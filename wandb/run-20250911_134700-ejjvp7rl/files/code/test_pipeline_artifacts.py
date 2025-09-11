#!/usr/bin/env python3
"""
模拟Woven pipeline输出的W&B artifact测试
专门复现pipeline输出artifacts丢失的问题
"""

import os
import sys
import time
import wandb
import tempfile
import json
from pathlib import Path

def create_pipeline_outputs():
    """创建模拟的pipeline输出文件"""
    temp_dir = Path(tempfile.mkdtemp(prefix="pipeline_outputs_"))
    
    # 1. 创建.pb文件（protobuf模型文件）
    pb_file = temp_dir / "model_v1.pb"
    with open(pb_file, 'wb') as f:
        # 模拟二进制protobuf数据
        fake_pb_data = b'\x08\x96\x01\x12\x04\x08\x01\x10\x01' * 1000  # 模拟protobuf格式
        f.write(fake_pb_data)
    
    # 2. 创建大的.pb文件（模拟25MB+的文件）
    large_pb_file = temp_dir / "large_model.pb"
    with open(large_pb_file, 'wb') as f:
        # 创建约5MB的文件（避免太大影响测试速度）
        chunk = b'\x08\x96\x01\x12\x04\x08\x01\x10\x01' * 10000
        for _ in range(50):  # 50 * 100KB ≈ 5MB
            f.write(chunk)
    
    # 3. 创建配置文件
    config_file = temp_dir / "pipeline_config.json"
    config_data = {
        "model_version": "v1.15.0",
        "pipeline_type": "demo_day_2023_few_s1",
        "created_at": time.time(),
        "triton_client": True,
        "inference_config": {
            "batch_size": 32,
            "max_sequence_length": 512
        }
    }
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # 4. 创建评估结果文件
    results_file = temp_dir / "evaluation_results.txt"
    with open(results_file, 'w') as f:
        f.write("MTMC Evaluation Results\n")
        f.write("======================\n")
        f.write("HOTA: 65.63\n")
        f.write("DetA: 68.05\n")
        f.write("AssA: 63.423\n")
        f.write("DetRe: 83.573\n")
        f.write("DetPr: 74.746\n")
        f.write("AssRe: 65.567\n")
        f.write("AssPr: 91.818\n")
        f.write("LocA: 89.072\n")
        f.write("OWTA: 72.796\n")
        f.write("HOTA(0): 74.894\n")
        f.write("LocA(0): 87.491\n")
    
    # 5. 创建多个输出文件（模拟复杂pipeline）
    outputs_dir = temp_dir / "outputs"
    outputs_dir.mkdir()
    
    for i in range(5):
        output_file = outputs_dir / f"output_{i:03d}.pb"
        with open(output_file, 'wb') as f:
            f.write(b'\x08\x96\x01\x12\x04\x08\x01\x10\x01' * (100 * (i + 1)))
    
    return temp_dir, [pb_file, large_pb_file, config_file, results_file] + list(outputs_dir.glob("*.pb"))

def main():
    print("=== Pipeline Artifacts 测试 ===")
    
    # 检查环境
    api_key = os.getenv('WANDB_API_KEY')
    base_url = os.getenv('WANDB_BASE_URL', 'https://api.wandb.ai')
    is_ci = os.getenv('CI', 'false').lower() == 'true'
    is_gha = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
    
    print(f"🔑 API Key: {'设置' if api_key else '未设置'}")
    print(f"🌐 Base URL: {base_url}")
    print(f"🏗️ CI环境: {'是' if is_ci else '否'}")
    print(f"🐙 GitHub Actions: {'是' if is_gha else '否'}")
    print(f"🐍 Python版本: {sys.version}")
    print(f"📦 W&B版本: {wandb.__version__}")
    
    if not api_key:
        print("❌ 错误: WANDB_API_KEY 未设置")
        return False
    
    try:
        # 初始化W&B - 模拟pipeline运行
        print("\n--- 初始化Pipeline运行 ---")
        run = wandb.init(
            project="pipeline-artifact-test",
            name=f"{'gha' if is_gha else 'local'}-pipeline-{int(time.time())}",
            tags=["pipeline-test", "artifact-debug", "demo_day_2023_few_s1"],
            config={
                "model_version": "v1.15.0",
                "pipeline_type": "demo_day_2023_few_s1",
                "environment": "gha" if is_gha else "local"
            }
        )
        print(f"✅ Pipeline运行初始化: {run.name}")
        print(f"✅ 项目: {run.project}")
        print(f"✅ URL: {run.url}")
        
        # 记录pipeline指标
        print("\n--- 记录Pipeline指标 ---")
        wandb.log({
            "pipeline_stage": "preprocessing",
            "processed_files": 1250,
            "processing_time": 45.2
        })
        wandb.log({
            "pipeline_stage": "inference", 
            "batch_size": 32,
            "inference_time": 120.5
        })
        wandb.log({
            "pipeline_stage": "evaluation",
            "mtmc_hota": 65.63,
            "mtmc_deta": 68.05,
            "mtmc_assa": 63.423
        })
        print("✅ Pipeline指标记录完成")
        
        # 创建pipeline输出文件
        print("\n--- 创建Pipeline输出 ---")
        temp_dir, pipeline_files = create_pipeline_outputs()
        
        total_size = sum(f.stat().st_size for f in pipeline_files)
        print(f"✅ 创建了 {len(pipeline_files)} 个pipeline文件")
        print(f"✅ 总大小: {total_size / (1024*1024):.1f} MB")
        
        for file in pipeline_files:
            size_mb = file.stat().st_size / (1024*1024)
            print(f"  📄 {file.name}: {size_mb:.1f} MB")
        
        # 创建pipeline输出artifact - 这是关键测试点
        print("\n--- 创建Pipeline输出Artifact ---")
        pipeline_artifact = wandb.Artifact(
            name="outputs-demo_day_2023_few_s1",  # 使用Woven的命名格式
            type="pipeline_output",  # 使用pipeline类型
            description="Pipeline输出文件 - 模拟Woven的demo_day_2023_few_s1"
        )
        
        # 添加所有pipeline文件
        for file in pipeline_files:
            pipeline_artifact.add_file(str(file))
            print(f"✅ 添加pipeline文件: {file.name}")
        
        # 上传pipeline artifact - 关键测试点
        print("\n--- 上传Pipeline Artifact ---")
        print("🔄 开始上传pipeline输出...")
        logged_artifact = wandb.log_artifact(pipeline_artifact)
        print("✅ log_artifact() 调用完成")
        
        # 等待上传完成
        print("🔄 等待pipeline artifact上传完成...")
        logged_artifact.wait()
        print(f"✅ Pipeline Artifact上传完成: {logged_artifact.name}:{logged_artifact.version}")
        
        # 额外等待时间（模拟Woven的30秒等待）
        print("\n--- Pipeline处理等待时间 ---")
        print("🔄 等待30秒确保pipeline后台处理完成...")
        time.sleep(30)
        
        # 通过API验证pipeline artifact
        print("\n--- 验证Pipeline Artifact ---")
        try:
            api = wandb.Api()
            retrieved_artifact = api.artifact(f"{run.entity}/{run.project}/outputs-demo_day_2023_few_s1:latest")
            print(f"✅ API验证成功: {retrieved_artifact.name}")
            print(f"✅ 文件数量: {len(retrieved_artifact.files())}")
            
            total_size = 0
            for file in retrieved_artifact.files():
                total_size += file.size
                print(f"  📄 {file.name} ({file.size / (1024*1024):.1f} MB)")
            
            print(f"✅ 总大小: {total_size / (1024*1024):.1f} MB")
                
        except Exception as e:
            print(f"❌ Pipeline Artifact API验证失败: {e}")
            print("🚨 这可能就是Woven遇到的问题！")
        
        # 清理临时文件
        import shutil
        shutil.rmtree(temp_dir)
        
        # 完成运行
        wandb.finish()
        print("\n✅ Pipeline测试完成")
        
        print(f"\n🔗 请检查W&B Dashboard: {run.url}")
        print("📋 重点检查:")
        print("  1. 运行是否出现在项目列表中")
        print("  2. Pipeline输出Artifact是否在Artifacts页面可见")
        print("  3. 所有.pb文件是否都在artifact中")
        print("  4. 大文件是否正确上传")
        print("  5. 对比staging环境和Woven环境的差异")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
