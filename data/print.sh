#!/bin/bash

# 定义目录列表
directories=("20030228_easy_ham" "20030228_easy_ham_2" "20030228_hard_ham" "20030228_spam" "20050311_spam_2")

# 遍历每个目录
for dir in "${directories[@]}"; do
    echo "Directory: $dir"
    echo "----------------------------------------"

    # 遍历每个子文件夹
    for subdir in "$dir"/*; do
        if [ -d "$subdir" ]; then
            echo "Subdirectory: $subdir"
            echo "Files:"

            # 获取子文件夹下的所有文件
            files=("$subdir"/*)

            # 输出前三个文件名
            for ((i = 0; i < 3 && i < ${#files[@]}; i++)); do
                echo "  ${files[$i]}"
            done

            # 输出后三个文件名
            for ((i = ${#files[@]} - 3; i < ${#files[@]}; i++)); do
                if ((i >= 0)); then
                    echo "  ${files[$i]}"
                fi
            done

            echo "----------------------------------------"
        fi
    done
done
