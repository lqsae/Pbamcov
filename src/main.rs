use std::time::Instant;
use std::{
    borrow::{Borrow, Cow},
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Error, Read, Result},
    path::Path,
    sync::{Arc, Mutex},
    thread,
};

use std::collections::BinaryHeap;
use std::io::{Seek, SeekFrom};
use std::iter::once;

use d4::{
    find_tracks,
    ssio::D4TrackReader,
    Chrom,
};
use d4_framefile::{Directory, OpenResult};

use rand::Rng;
use regex::Regex;
use clap::{Arg, App};
use rayon::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use itertools::Itertools;

// 定义数据结构
#[derive(Debug, Clone, Copy)]
struct Range {
    length: u32,
    value: u16
}

#[derive(Default)]
struct SummaryStats {
    x1: u64,
    x10: u64,
    x20: u64,
    x30: u64,
    x50: u64,
    variance_sum: f64,
    large_avg_20: u64,
    q20: f64,
}

#[derive(Default)]
struct CovStatResult {
    stats: SummaryStats,
}

// 解析区域规格
fn parse_region_spec<T: Iterator<Item = String>>(
    regions: Option<T>,
    chrom_list: &[Chrom],
) -> std::io::Result<Vec<(usize, u32, u32)>> {
    let region_pattern = Regex::new(r"^(?P<CHR>[^:]+)((:(?P<FROM>\d+)-)?(?P<TO>\d+)?)?$").unwrap();
    let chr_map: HashMap<_, _> = chrom_list
        .iter()
        .enumerate()
        .map(|(a, b)| (b.name.to_string(), a))
        .collect();

    Ok(regions.map_or_else(
        || chrom_list.iter().enumerate().map(|(id, chrom)| (id, 0, chrom.size as u32)).collect(),
        |regions| regions.filter_map(|region_spec| {
            region_pattern.captures(&region_spec).and_then(|captures| {
                let chr = captures.name("CHR").unwrap().as_str();
                let start: u32 = captures.name("FROM").map_or(0, |x| x.as_str().parse().unwrap_or(0));
                let end: u32 = captures.name("TO").map_or_else(
                    || chr_map.get(chr).map_or(!0, |&id| chrom_list[id].size as u32),
                    |x| x.as_str().parse().unwrap_or(!0)
                );
                chr_map.get(chr).map(|&chr| (chr, start, end))
            }).or_else(|| {
                eprintln!("警告：忽略未在d4文件中定义的染色体 {}", region_spec);
                None
            })
        }).collect()
    ))
}

// 添加单个区间的统计信息
fn add_one_interval(
    length: u32,
    values: &[i32],
    vect: &mut Vec<Range>,
    all_length: &mut u64,
    totaldepth: &mut u64,
    depthx1: &mut u64
) {
    *all_length += length as u64;
    
    for &value in values {
        let value_u32 = value as u32;
        *totaldepth += (length as u64) * (value_u32 as u64);
        if value_u32 >= 1 {
            *depthx1 += length as u64;
        }
        vect.push(Range { length, value: value as u16 });
    }
}

// 添加多个区间的统计信息
fn add_some_interval<R: Read + Seek>(
    inputs: &mut [D4TrackReader<R>],
    regions: &[(usize, u32, u32)],
    vect: &mut Vec<Range>,
    all_length: &mut u64,
    totaldepth: &mut u64,
    depthx1: &mut u64
) -> u32 {
    if inputs.is_empty() {
        return 1;
    }
    
    for &(cid, begin, end) in regions {
        let chrom = inputs[0].chrom_list()[cid].name.as_str().to_string();
        let mut values = vec![0; inputs.len()];
        let mut prev_values = vec![0; inputs.len()];
        let mut views: Vec<_> = inputs
            .iter_mut()
            .map(|x| x.get_view(&chrom, begin, end).unwrap())
            .collect();

        let mut value_changed = false;
        let mut last_pos = begin;

        for pos in begin..end {
            for (input_id, input) in views.iter_mut().enumerate() {
                if let Ok((reported_pos, value)) = input.next().unwrap() {
                    debug_assert_eq!(reported_pos, pos);
                    if values[input_id] != value {
                        if !value_changed {
                            prev_values.copy_from_slice(&values);
                            value_changed = true;
                        }
                        values[input_id] = value;
                    }
                }
            }
            if value_changed {
                add_one_interval(pos - last_pos, &prev_values, vect, all_length, totaldepth, depthx1);
                last_pos = pos;
                value_changed = false;
            }
        }
        if last_pos != end {
            add_one_interval(end - last_pos, &prev_values, vect, all_length, totaldepth, depthx1);
        }
    }
    0
}

// 添加多个区间的统计信息，支持从文件读取
fn add_ranges <R: Read + Seek, I: Iterator<Item = String>> (  
    mut reader: R,
    pattern: &Regex,
    track: Option<&str>,
    regions: Option<I>,
    vect: & mut Vec<Range>,
    all_length: &mut u64,
    totaldepth: &mut u64, 
    depthx1: &mut u64)  -> u32 {
    let first = false;
    let mut path_buf = vec![];
    let mut first_found = false;
    if let Some(track_path) = track {
        path_buf.push(track_path.into());
    } else {
        find_tracks(
            &mut reader,
            |path| {
                let stem = path
                    .map(|what: &Path| {
                        what.file_name()
                            .map(|x| x.to_string_lossy())
                            .unwrap_or_else(|| Cow::<str>::Borrowed(""))
                    })
                    .unwrap_or_default();
                if pattern.is_match(stem.borrow()) {
                    if first && first_found {
                        false
                    } else {
                        first_found = true;
                        true
                    }
                } else {
                    false
                }
            },
            &mut path_buf,
        );
    }
    let file_root = Directory::open_root(reader, 8).unwrap();
    let mut readers = vec![];
    for path in path_buf.iter() {
        let track_root = match file_root.open(path).unwrap() {
            OpenResult::SubDir(track_root) => track_root,
            _ => {
                return 0;
            }
        };
        let reader = D4TrackReader::from_track_root(track_root).unwrap();
        readers.push(reader);
    }
    let regions = parse_region_spec(regions, readers[0].chrom_list()).unwrap();
    add_some_interval(& mut readers, &regions, vect, all_length, totaldepth, depthx1);
    0
}

// 从文件读取区域信息
fn read_regions(region_file: &str) -> Vec<String> {
    BufReader::new(File::open(region_file).unwrap())
        .lines()
        .filter_map(|line| {
            let line = line.unwrap();
            if line.starts_with('#') {
                return None;
            }
            let mut splitted = line.trim().split('\t');
            match (splitted.next(), splitted.next(), splitted.next()) {
                (Some(chr), Some(beg), Some(end)) => {
                    beg.parse::<u32>().ok().and_then(|begin| {
                        end.parse::<u32>().ok().map(|end| {
                            format!("{}:{}-{}", chr, begin, end)
                        })
                    })
                },
                _ => None
            }
        })
        .collect()
}

// 排序获取分位数
// fn calculate_quantile(interval_depth: &Vec<Range>, quantile: f64, total_length: & mut u64) -> u16
// {
//     //对interval_depth  
//     let mut sorted_depths = interval_depth.to_vec();
//     sorted_depths.par_sort_unstable_by(|a, b| b.value.cmp(&a.value));

//     let mut idx: u64 = 0;
//     let mut idx_q20 = (*total_length as f64 * quantile)  as u64;
//     let mut jixu: bool = true;
//     let mut  q20 :u16 = 0;
//     for range_value in sorted_depths.iter()
//     {
//         let value = range_value.value;
//         let length = range_value.length as u64;
//         if value >=1 {idx += length;}
//         if jixu {
//             if idx >= idx_q20{
//                 q20 = value; 
//                 jixu =false;}
//         }
//         else {
//             break;
//         } 
//     }   
//     q20
// }


//基于快速选择算法计算分位数
// fn calculate_quantile(interval_depth: &[Range], quantile: f64, total_length: &mut u64) -> u16 {
//     let target_index = (*total_length as f64 * quantile) as u64;
    
//     fn weighted_quick_select(depths: &[Range], k: u64) -> u16 {
//         if depths.len() == 1 {
//             return depths[0].value;
//         }

//         let pivot = *depths.choose(&mut thread_rng()).unwrap();
//         let (mut left, mut equal, mut right): (Vec<Range>, Vec<Range>, Vec<Range>) = (Vec::new(), Vec::new(), Vec::new());

//         for range in depths {
//             if range.value > pivot.value {
//                 left.push(range.clone());
//             } else if range.value < pivot.value {
//                 right.push(range.clone());
//             } else {
//                 equal.push(range.clone());
//             }
//         }

//         let left_count: u64 = left.iter().map(|r| r.length as u64).sum();
//         let equal_count: u64 = equal.iter().map(|r| r.length as u64).sum();

//         if k < left_count {
//             weighted_quick_select(&left, k)
//         } else if k < left_count + equal_count {
//             pivot.value
//         } else {
//             weighted_quick_select(&right, k - left_count - equal_count)
//         }
//     }

//     weighted_quick_select(interval_depth, target_index)
// }



// 计算汇总统计信息
fn calculate_summary_stats(interval_depth: &[Range], mean: f64) -> SummaryStats {
    let mut stats = interval_depth.par_iter().map(|range| {
        let depth = range.value as f64;
        let length = range.length as u64;
        SummaryStats {
            x1: if depth >= 1.0 { length } else { 0 },
            x10: if depth >= 10.0 { length } else { 0 },
            x20: if depth >= 20.0 { length } else { 0 },
            x30: if depth >= 30.0 { length } else { 0 },
            x50: if depth >= 50.0 { length } else { 0 },
            variance_sum: length as f64 * (depth - mean).powi(2),
            large_avg_20: if depth >= 0.2 * mean { length } else { 0 },
            q20: 0.0, // 初始化为0，稍后计算
        }
    }).reduce(|| SummaryStats::default(),
               |a, b| SummaryStats {
                   x1: a.x1 + b.x1,
                   x10: a.x10 + b.x10,
                   x20: a.x20 + b.x20,
                   x30: a.x30 + b.x30,
                   x50: a.x50 + b.x50,
                   variance_sum: a.variance_sum + b.variance_sum,
                   large_avg_20: a.large_avg_20 + b.large_avg_20,
                   q20: 0.0, // 在这里保持为0
               });
    stats
}


// 基于并行版本的快速选择算法计算分位数
fn calculate_quantile(interval_depth: &[Range], quantile: f64, total_length: &mut u64) -> u16 {
    let target_index = (*total_length as f64 * quantile) as u64;
    
    fn weighted_quick_select(depths: &[Range], k: u64) -> u16 {
        if depths.len() == 1 {
            return depths[0].value;
        }
        let pivot = *depths.choose(&mut thread_rng()).unwrap();
        let (left, equal, right): (Vec<Range>, Vec<Range>, Vec<Range>) = depths.par_iter()
            .fold(
                || (Vec::new(), Vec::new(), Vec::new()),
                |(mut l, mut e, mut r), range| {
                    if range.value > pivot.value {
                        l.push(range.clone());
                    } else if range.value < pivot.value {
                        r.push(range.clone());
                    } else {
                        e.push(range.clone());
                    }
                    (l, e, r)
                }
            )
            .reduce(
                || (Vec::new(), Vec::new(), Vec::new()),
                |mut a, b| {
                    a.0.extend(b.0);
                    a.1.extend(b.1);
                    a.2.extend(b.2);
                    a
                }
            );
        let left_count: u64 = left.par_iter().map(|r| r.length as u64).sum();
        let equal_count: u64 = equal.par_iter().map(|r| r.length as u64).sum();
        if k < left_count {
            weighted_quick_select(&left, k)
        } else if k < left_count + equal_count {
            pivot.value
        } else {
            weighted_quick_select(&right, k - left_count - equal_count)
        }
    }

    weighted_quick_select(interval_depth, target_index)
}

// 主函数
fn main() {
    
    let start_time = Instant::now();
    // 计算最终统计数据
    let args = App::new("BamCov")
        .version("0.1")
        .author("liuqingshan")
        .about("基于mosdepth d4文件计算WGS、外显子组或靶向测序的碱基覆盖")
        .arg(Arg::with_name("d4-file")
            .short("d")
            .long("d4-format")
            .takes_value(true)
            .required(true)
            .help("d4文件 https://github.com/38/d4-format"))
        .arg(Arg::with_name("region-file")
            .long("region")
            .short("r")
            .takes_value(true)
            .required(true)
            .help("输入为bed文件格式"))
        .arg(Arg::with_name("threads")
            .short("t")
            .long("threads")
            .takes_value(true)
            .default_value("4")
            .help("使用的线程数"))
        .get_matches();

    let input_filename = args.value_of("d4-file").unwrap();
    let region_file = args.value_of("region-file").unwrap();
    let thread_count: usize = args.value_of("threads").unwrap().parse().expect("无效的线程数");

    // 读取区域信息并进行多线程处理
    let regions = Arc::new(read_regions(region_file));

    // 确保线程数不超过区域数量
    let effective_thread_count = std::cmp::min(thread_count, regions.len());
    
    if effective_thread_count == 0 {
        println!("警告：没有需要处理的区域");
        return;
    }
    let chunk_size = (regions.len() + effective_thread_count - 1) / effective_thread_count;

    // 创建共享的数据结构
    let vect = Arc::new(Mutex::new(Vec::new()));
    let all_length = Arc::new(Mutex::new(0u64));
    let totaldepth = Arc::new(Mutex::new(0u64));
    let depthx1 = Arc::new(Mutex::new(0u64));

//    // 在线程池外创建 reader 和 track_pattern
//     let reader = Arc::new(Mutex::new(File::open(&input_filename).unwrap()));
//     let track_pattern = Arc::new(Regex::new(".*").unwrap());


    // 创建线程并分配任务
    let handles: Vec<_> = (0..effective_thread_count).map(|i| {
        // 计算每个线程处理的区域范围
        let start = i * chunk_size;
        let end = std::cmp::min((i + 1) * chunk_size, regions.len());
        
        // 克隆共享数据的引用
        let regions = Arc::clone(&regions);
        let vect = Arc::clone(&vect);
        let all_length = Arc::clone(&all_length);
        let totaldepth = Arc::clone(&totaldepth);
        let depthx1 = Arc::clone(&depthx1);
        let input_filename = input_filename.to_string();

        // 创建新线程
        thread::spawn(move || {
            // 只有当 start < end 时才处理区域
            if start < end {
                // 初始化局部变量
                let mut local_vect = Vec::new();
                let mut local_all_length = 0u64;
                let mut local_totaldepth = 0u64;
                let mut local_depthx1 = 0u64;

                // 打开输入文件
                let mut reader = File::open(&input_filename).unwrap();
                let track_pattern = Regex::new(".*").unwrap();
                // 处理分配给该线程的区域
                for region in &regions[start..end] {
                    add_ranges(
                        &mut reader,
                        &track_pattern,
                        None,
                        Some(std::iter::once(region.clone())),
                        &mut local_vect,
                        &mut local_all_length,
                        &mut local_totaldepth,
                        &mut local_depthx1,
                    );
                    reader.rewind().unwrap();
                }
                // 将局部结果合并到共享数据中
                let mut vect = vect.lock().unwrap();
                vect.extend(local_vect);
                let mut all_length = all_length.lock().unwrap();
                *all_length += local_all_length;
                let mut totaldepth = totaldepth.lock().unwrap();
                *totaldepth += local_totaldepth;
                let mut depthx1 = depthx1.lock().unwrap();
                *depthx1 += local_depthx1;
            }
        })
    }).collect();

    // 等待所有线程完成 
    for handle in handles {
        handle.join().unwrap();
    }
    //统计运行时间
    let elapsed_time = start_time.elapsed();
    println!("获取测序深度运行时间: {:?}", elapsed_time);  

    // 将共享数据转换为局部变量并进行统计计算
    let vect = Arc::try_unwrap(vect).unwrap().into_inner().unwrap();
    let mut all_length = Arc::try_unwrap(all_length).unwrap().into_inner().unwrap();
    let totaldepth = Arc::try_unwrap(totaldepth).unwrap().into_inner().unwrap();
    let depthx1 = Arc::try_unwrap(depthx1).unwrap().into_inner().unwrap();
    let mean = totaldepth as f64 / all_length as f64;

    //统计运行时间
    let start_time_calculate_summary_stats = Instant::now();
    let mut  stats = calculate_summary_stats(&vect, mean);
    let elapsed_time_calculate_summary_stats = start_time_calculate_summary_stats.elapsed();
    println!("calculate_summary_stats 运行时间: {:?}", elapsed_time_calculate_summary_stats); 
    
    let start_time_q20 = Instant::now();
    let q20 = calculate_quantile(&vect, 0.2, &mut all_length);
    let elapsed_time_q20 = start_time_q20.elapsed();
    println!("q20 运行时间: {:?}", elapsed_time_q20); 
    
    stats.q20 = q20 as f64;
    let q20_value =  stats.q20;
    let all_length_f64 = all_length as f64;
    let variance = stats.variance_sum / all_length_f64;
    let std_deviation = variance.sqrt();
    let x1_cov = (stats.x1 as f64) / all_length_f64 * 100.0;
    let x10_cov = (stats.x10 as f64) / all_length_f64 * 100.0;
    let x20_cov = (stats.x20 as f64) / all_length_f64 * 100.0;
    let x30_cov = (stats.x30 as f64) / all_length_f64 * 100.0;
    let x50_cov = (stats.x50 as f64) / all_length_f64 * 100.0;
    let cv = std_deviation / mean;
    let fold80 = q20_value / mean;
    let q20_cov = (stats.large_avg_20 as f64) / all_length_f64 * 100.0;
    // 输出结果
    println!("TotalBases\tCovBases\tCovRatio\tAve_Depth(X)\tDepth>=1X\tDepth>=10X\tDepth>=20X\tDepth>=30X\tDepth>=50X\tFold80\tCV\t>=20%X");
    println!("{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}", 
             all_length, stats.x1, x1_cov, mean, x1_cov, x10_cov, x20_cov, x30_cov, x50_cov, fold80, cv, q20_cov);
    let elapsed_time = start_time.elapsed();
    println!("总执行时间: {:?}", elapsed_time);
}

