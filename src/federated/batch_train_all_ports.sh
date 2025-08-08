#!/bin/bash

# GAT-FedPPO 批量训练脚本
# 用法: ./batch_train_all_ports.sh [选项]

set -e  # 遇到错误立即退出

# 默认配置
PORTS=("baton_rouge" "new_orleans" "south_louisiana" "gulfport")
LOG_DIR="../../logs/batch_training"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PARALLEL=false
CONSISTENCY_TEST=true

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL=true
            shift
            ;;
        --no-consistency-test)
            CONSISTENCY_TEST=false
            shift
            ;;
        --ports)
            IFS=',' read -ra PORTS <<< "$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --parallel              并行训练所有港口"
            echo "  --no-consistency-test   跳过一致性测试"
            echo "  --ports PORT1,PORT2     指定要训练的港口（逗号分隔）"
            echo "  --log-dir DIR           指定日志目录"
            echo "  -h, --help              显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                                    # 顺序训练所有港口"
            echo "  $0 --parallel                        # 并行训练所有港口"
            echo "  $0 --ports baton_rouge,gulfport      # 只训练指定港口"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 创建日志目录
mkdir -p "$LOG_DIR"

echo "🚀 开始批量训练 GAT-FedPPO"
echo "📅 时间戳: $TIMESTAMP"
echo "🏗️ 训练港口: ${PORTS[*]}"
echo "📝 日志目录: $LOG_DIR"
echo "⚡ 并行模式: $PARALLEL"
echo "🔍 一致性测试: $CONSISTENCY_TEST"
echo ""

# 训练函数
train_port() {
    local port=$1
    local log_file="$LOG_DIR/train_${port}_${TIMESTAMP}.log"
    
    echo "🏗️ 开始训练港口: $port"
    echo "📝 日志文件: $log_file"
    
    if python curriculum_trainer.py --port "$port" > "$log_file" 2>&1; then
        echo "✅ 港口 $port 训练成功"
        return 0
    else
        echo "❌ 港口 $port 训练失败，查看日志: $log_file"
        return 1
    fi
}

# 记录开始时间
start_time=$(date +%s)

# 训练所有港口
failed_ports=()

if [ "$PARALLEL" = true ]; then
    echo "⚡ 并行训练模式"
    pids=()
    
    for port in "${PORTS[@]}"; do
        train_port "$port" &
        pids+=($!)
    done
    
    # 等待所有进程完成
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            echo "✅ 进程 ${pids[$i]} (${PORTS[$i]}) 完成"
        else
            echo "❌ 进程 ${pids[$i]} (${PORTS[$i]}) 失败"
            failed_ports+=("${PORTS[$i]}")
        fi
    done
else
    echo "🔄 顺序训练模式"
    for port in "${PORTS[@]}"; do
        if ! train_port "$port"; then
            failed_ports+=("$port")
        fi
        echo ""
    done
fi

# 计算训练时间
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo ""
echo "⏱️ 训练完成，总耗时: ${hours}h ${minutes}m ${seconds}s"

# 训练结果总结
total_ports=${#PORTS[@]}
failed_count=${#failed_ports[@]}
success_count=$((total_ports - failed_count))

echo ""
echo "📊 训练结果总结:"
echo "  总港口数: $total_ports"
echo "  成功数: $success_count"
echo "  失败数: $failed_count"

if [ $failed_count -eq 0 ]; then
    echo "🎉 所有港口训练成功！"
else
    echo "⚠️ 失败的港口: ${failed_ports[*]}"
fi

# 运行一致性测试
if [ "$CONSISTENCY_TEST" = true ] && [ $failed_count -eq 0 ]; then
    echo ""
    echo "🔍 开始一致性测试..."
    
    consistency_log="$LOG_DIR/consistency_test_${TIMESTAMP}.log"
    if python consistency_test.py --samples 200 > "$consistency_log" 2>&1; then
        echo "✅ 一致性测试完成，日志: $consistency_log"
    else
        echo "❌ 一致性测试失败，日志: $consistency_log"
    fi
fi

# 生成训练报告
report_file="$LOG_DIR/batch_training_report_${TIMESTAMP}.md"
cat > "$report_file" << EOF
# 批量训练报告

**训练时间**: $(date -d @$start_time '+%Y-%m-%d %H:%M:%S') - $(date -d @$end_time '+%Y-%m-%d %H:%M:%S')
**总耗时**: ${hours}h ${minutes}m ${seconds}s
**训练模式**: $([ "$PARALLEL" = true ] && echo "并行" || echo "顺序")

## 训练结果

- **总港口数**: $total_ports
- **成功数**: $success_count
- **失败数**: $failed_count
- **成功率**: $(( success_count * 100 / total_ports ))%

## 港口详情

EOF

for port in "${PORTS[@]}"; do
    if [[ " ${failed_ports[*]} " =~ " $port " ]]; then
        echo "- **$port**: ❌ 失败" >> "$report_file"
    else
        echo "- **$port**: ✅ 成功" >> "$report_file"
    fi
done

cat >> "$report_file" << EOF

## 日志文件

EOF

for port in "${PORTS[@]}"; do
    log_file="train_${port}_${TIMESTAMP}.log"
    echo "- **$port**: \`$log_file\`" >> "$report_file"
done

if [ "$CONSISTENCY_TEST" = true ]; then
    echo "- **一致性测试**: \`consistency_test_${TIMESTAMP}.log\`" >> "$report_file"
fi

echo ""
echo "📋 训练报告已生成: $report_file"

# 退出码
if [ $failed_count -eq 0 ]; then
    exit 0
else
    exit 1
fi