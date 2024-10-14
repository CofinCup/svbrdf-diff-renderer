#!/bin/bash

# 定义你的test_type列表和GPU列表
json_paths=("/root/datasets/montage/multi-pics-with-noise.json" \
            "/root/datasets/montage/multi-pics-with-noise_v2.json" \
            "/root/datasets/montage/multi-pics_noiseless.json" \
            "/root/datasets/montage/multi-pics_noiseless_replace_normal.json" \
            "/root/datasets/montage/multi-pics_noiseless_with_normal.json" \
     )  # 根据你的json路径修改
result_paths=("/root/test_data/materialgan/nonspecular-2000epoch/multi-pics-with-noise" \
              "/root/test_data/materialgan/nonspecular-2000epoch/multi-pics-with-noise_v2" \
              "/root/test_data/materialgan/nonspecular-2000epoch/multi-pics_noiseless" \
              "/root/test_data/materialgan/nonspecular-2000epoch/multi-pics_noiseless_replace_normal" \
              "/root/test_data/materialgan/nonspecular-2000epoch/multi-pics_noiseless_with_normal" \
     )  # 根据你的结果路径修改
test_types=("focus" "random" "uniform")
GPULists=("0,1,2,3" "1,2,3,0" "2,3,0,1" "3,0,1,2")  # 根据你的GPU列表修改

# 初始化GPU计数器
gpu_count=${#GPULists[@]}

# 用于存储后台进程的PID
pids=()

# 定义退出时的清理操作
cleanup() {
    echo "清理中...强制关闭后台程序。"
    for pid in "${pids[@]}"; do
        kill -9 "$pid" 2>/dev/null
    done
    exit
}

# 捕获退出信号
trap cleanup EXIT

# 遍历json_paths和test_types
for json_index in "${!json_paths[@]}"; do
  json_path=${json_paths[$json_index]}
  # 提取json文件名，不包括路径
  json_filename=$(basename "$json_path" .json)

  for test_index in "${!test_types[@]}"; do
    test_type=${test_types[$test_index]}

    # 通过循环分配GPU，保证GPU轮询使用
    gpu=${GPULists[$(((json_index * ${#test_types[@]} + test_index) % gpu_count))]}

    # 输出当前使用的配置
    echo "正在处理: json_path=${json_path}, test_type=${test_type}, 使用GPU=${gpu}"

    # 创建唯一的日志文件用于保存输出
    log_file="/root/test_data/materialgan/nonspecular-2000epoch/output_${json_filename}_${test_type}.log"
    temp_log_file="${log_file}.tmp"

    # 在后台运行Python脚本，并将输出重定向到临时日志文件
    {
      python full_run.py --json_path "$json_path" --test_type "$test_type" --gpu "$gpu" --result_path "${result_paths[$json_index]}"
      exit_code=$?
      if [ $exit_code -ne 0 ]; then
        echo "错误：脚本在处理 ${json_filename} 的 ${test_type} 时异常退出，退出状态码: ${exit_code}" >> "$temp_log_file"
      fi
    } > "$temp_log_file" 2>&1 &

    # 将后台进程的PID添加到数组
    pids+=("$!")

    # 定期检查并裁剪临时日志文件
    {
      while sleep 10; do
        if [ -f "$temp_log_file" ] && [ $(stat --format=%s "$temp_log_file") -gt 1048576 ]; then
          echo "日志文件 ${temp_log_file} 超过1MB，正在裁剪..." >> "$temp_log_file"
          mv "$temp_log_file" "$log_file"
          truncate -s 1M "$log_file"
        fi
      done
    } &
  done
done

# 等待所有后台进程完成
for pid in "${pids[@]}"; do
    wait "$pid" || {
        echo "后台进程 $pid 发生错误，查看日志文件以获取详细信息。"
    }
done

echo "所有进程已完成。"
