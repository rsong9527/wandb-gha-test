#!/usr/bin/env python3
"""
极端Pipeline测试 - 尝试复现Woven的具体问题
模拟更复杂的场景：大量文件、复杂的artifact结构、并发上传等
"""

import os
import sys
import time
import wandb
import tempfile
import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def create_complex_pipeline_outputs():
    """创建复杂的pipeline输出文件结构"""
    temp_dir = Path(tempfile.mkdtemp(prefix="complex_pipeline_"))
    
    files = []
    
    # 1. 创建多个大的.pb文件（模拟Triton模型）
    for i in range(3):
        pb_file = temp_dir / f"triton_model_{i:02d}.pb"
        with open(pb_file, 'wb') as f:
            # 创建约2MB的文件
            chunk = b'\x08\x96\x01\x12\x04\x08\x01\x10\x01' * 1000
            for _ in range(20):  # 20 * 100KB ≈ 2MB
                f.write(chunk)
        files.append(pb_file)
    
    # 2. 创建嵌套目录结构（模拟复杂的输出）
    for dataset in ['demo_day_2023_few_s1', 'multiview_5_cams', 'combined']:
        dataset_dir = temp_dir / dataset
        dataset_dir.mkdir()
        
        # 评估结果文件
        results_file = dataset_dir / f"{dataset}_results.json"
        results_data = {
            "dataset": dataset,
            "metrics": {
                "HOTA": 65.63 + (hash(dataset) % 100) / 100,
                "DetA": 68.05 + (hash(dataset) % 100) / 100,
                "AssA": 63.423 + (hash(dataset) % 100) / 100,
                "DetRe": 83.573,
                "DetPr": 74.746,
                "AssRe": 65.567,
                "AssPr": 91.818,
                "LocA": 89.072,
                "OWTA": 72.796
            },
            "identity_metrics": {
                "IDF1": 70.281,
                "IDR": 74.431,
                "IDP": 66.57,
                "IDTP": 4582,
                "IDFN": 1574,
                "IDFP": 2301
            },
            "count_metrics": {
                "Dets": 6883,
                "GT_Dets": 6156,
                "IDs": 38,
                "GT_IDs": 8
            }
        }
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        files.append(results_file)
        
        # 创建输出文件
        for j in range(5):
            output_file = dataset_dir / f"output_{j:03d}.pb"
            with open(output_file, 'wb') as f:
                f.write(b'\x08\x96\x01\x12\x04\x08\x01\x10\x01' * (200 * (j + 1)))
            files.append(output_file)
    
    # 3. 创建配置和元数据文件
    config_file = temp_dir / "pipeline_config.yaml"
    with open(config_file, 'w') as f:
        f.write("""
pipeline:
  version: v1.15.0
  type: demo_day_2023_few_s1
  triton_client: true
  
datasets:
  - demo_day_2023_few_s1
  - multiview_5_cams
  - combined
  
inference:
  batch_size: 32
  max_sequence_length: 512
  model_path: /models/triton_model_00.pb
  
evaluation:
  metrics:
    - HOTA
    - DetA
    - AssA
    - MTMC
  thresholds: [0.5, 0.75, 0.9]
""")
    files.append(config_file)
    
    return temp_dir, files

def create_multiple_artifacts(run, temp_dir, files):
    """创建多个不同类型的artifacts"""
    artifacts = []
    
    # 1. Pipeline输出artifact（主要的）
    pipeline_artifact = wandb.Artifact(
        name="outputs-demo_day_2023_few_s1",
        type="pipeline_output",
        description="完整的pipeline输出 - 模拟Woven场景"
    )
    
    # 添加所有文件
    for file in files:
        pipeline_artifact.add_file(str(file))
    
    artifacts.append(("pipeline_output", pipeline_artifact))
    
    # 2. 模型artifact
    model_artifact = wandb.Artifact(
        name="triton-models",
        type="model",
        description="Triton推理模型"
    )
    
    for file in files:
        if file.name.endswith('.pb'):
            model_artifact.add_file(str(file))
    
    artifacts.append(("model", model_artifact))
    
    # 3. 评估结果artifact
    results_artifact = wandb.Artifact(
        name="evaluation-results",
        type="evaluation",
        description="MTMC评估结果"
    )
    
    for file in files:
        if file.name.endswith('.json'):
            results_artifact.add_file(str(file))
    
    artifacts.append(("evaluation", results_artifact))
    
    return artifacts

def upload_artifact_with_retry(artifact_info, max_retries=3):
    """带重试的artifact上传"""
    artifact_type, artifact = artifact_info
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 上传{artifact_type} artifact (尝试 {attempt + 1}/{max_retries})")
            logged_artifact = wandb.log_artifact(artifact)
            logged_artifact.wait()
            print(f"✅ {artifact_type} artifact上传成功: {logged_artifact.name}:{logged_artifact.version}")
            return logged_artifact
        except Exception as e:
            print(f"❌ {artifact_type} artifact上传失败 (尝试 {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                raise
            time.sleep(5)  # 等待5秒后重试

def main():
    print("=== 极端Pipeline测试 ===")
    
    # 检查环境
    api_key = os.getenv('WANDB_API_KEY')
    base_url = os.getenv('WANDB_BASE_URL', 'https://api.wandb.ai')
    is_ci = os.getenv('CI', 'false').lower() == 'true'
    is_gha = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
    
    print(f"🔑 API Key: {'设置' if api_key else '未设置'}")
    print(f"🌐 Base URL: {base_url}")
    print(f"🏗️ CI环境: {'是' if is_ci else '否'}")
    print(f"🐙 GitHub Actions: {'是' if is_gha else '否'}")
    
    if not api_key:
        print("❌ 错误: WANDB_API_KEY 未设置")
        return False
    
    try:
        # 初始化W&B
        print("\n--- 初始化极端Pipeline运行 ---")
        run = wandb.init(
            project="extreme-pipeline-test",
            name=f"{'gha' if is_gha else 'local'}-extreme-{int(time.time())}",
            tags=["extreme-test", "pipeline-stress", "artifact-debug"],
            config={
                "test_type": "extreme_pipeline",
                "environment": "gha" if is_gha else "local",
                "datasets": ["demo_day_2023_few_s1", "multiview_5_cams", "combined"],
                "model_count": 3,
                "total_files": "18+"
            }
        )
        print(f"✅ 极端Pipeline运行初始化: {run.name}")
        print(f"✅ URL: {run.url}")
        
        # 记录复杂的指标
        print("\n--- 记录复杂指标 ---")
        for stage in ["preprocessing", "inference", "evaluation"]:
            wandb.log({
                f"{stage}_time": 45.2 + hash(stage) % 100,
                f"{stage}_memory_mb": 1024 + hash(stage) % 2048,
                f"{stage}_success": True
            })
        
        # 记录MTMC指标
        for dataset in ["demo_day_2023_few_s1", "multiview_5_cams", "combined"]:
            wandb.log({
                f"mtmc_{dataset}_hota": 65.63 + (hash(dataset) % 100) / 100,
                f"mtmc_{dataset}_deta": 68.05 + (hash(dataset) % 100) / 100,
                f"mtmc_{dataset}_assa": 63.423 + (hash(dataset) % 100) / 100
            })
        
        print("✅ 复杂指标记录完成")
        
        # 创建复杂的pipeline输出
        print("\n--- 创建复杂Pipeline输出 ---")
        temp_dir, files = create_complex_pipeline_outputs()
        
        total_size = sum(f.stat().st_size for f in files)
        print(f"✅ 创建了 {len(files)} 个复杂文件")
        print(f"✅ 总大小: {total_size / (1024*1024):.1f} MB")
        
        # 创建多个artifacts
        print("\n--- 创建多个Artifacts ---")
        artifacts = create_multiple_artifacts(run, temp_dir, files)
        print(f"✅ 准备了 {len(artifacts)} 个不同类型的artifacts")
        
        # 并发上传artifacts（模拟高负载）
        print("\n--- 并发上传Artifacts ---")
        uploaded_artifacts = []
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(upload_artifact_with_retry, artifact_info) 
                      for artifact_info in artifacts]
            
            for future in futures:
                try:
                    logged_artifact = future.result(timeout=300)  # 5分钟超时
                    uploaded_artifacts.append(logged_artifact)
                except Exception as e:
                    print(f"❌ 并发上传失败: {e}")
        
        print(f"✅ 成功上传了 {len(uploaded_artifacts)} 个artifacts")
        
        # 极端等待时间（模拟Woven的处理时间）
        print("\n--- 极端等待时间 ---")
        print("🔄 等待60秒模拟复杂的后台处理...")
        time.sleep(60)
        
        # 验证所有artifacts
        print("\n--- 验证所有Artifacts ---")
        api = wandb.Api()
        
        for artifact_type, original_artifact in artifacts:
            try:
                artifact_name = original_artifact.name
                retrieved_artifact = api.artifact(f"{run.entity}/{run.project}/{artifact_name}:latest")
                print(f"✅ {artifact_type} API验证成功: {retrieved_artifact.name}")
                print(f"  📄 文件数量: {len(retrieved_artifact.files())}")
                
                total_size = sum(file.size for file in retrieved_artifact.files())
                print(f"  💾 总大小: {total_size / (1024*1024):.1f} MB")
                
            except Exception as e:
                print(f"❌ {artifact_type} API验证失败: {e}")
                print("🚨 这可能就是Woven遇到的问题！")
        
        # 清理
        import shutil
        shutil.rmtree(temp_dir)
        
        wandb.finish()
        print("\n✅ 极端Pipeline测试完成")
        
        print(f"\n🔗 请检查W&B Dashboard: {run.url}")
        print("📋 重点检查:")
        print("  1. 所有artifacts是否都显示在dashboard中")
        print("  2. 大文件和复杂结构是否正确处理")
        print("  3. 并发上传是否导致任何问题")
        print("  4. 对比本地和GHA环境的差异")
        
        return True
        
    except Exception as e:
        print(f"❌ 极端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


