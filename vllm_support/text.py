"""
快速测试脚本
验证所有组件是否正常工作
"""
import torch
import sys

def test_attention():
    """测试 PagedAttention"""
    print("测试 1: PagedAttention 模块...")
    try:
        from vllm_attention import PagedCausalMultiHeadAttention, PagedKVCache
        
        # 创建注意力模块
        attn = PagedCausalMultiHeadAttention(
            d_model=256,
            n_head=4,
            max_seq_size=512,
            num_kv_blocks=64,
            block_size=16,
            device='cpu'
        )
        
        # 测试前向传播
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, 256)
        
        # 训练模式
        output = attn(x, is_prefill=True, block_tables=None)
        assert output.shape == (batch_size, seq_len, 256)
        
        print("  ✓ 训练模式测试通过")
        
        # Prefill 模式
        block_tables = torch.randint(0, 64, (batch_size, 2))
        slot_mapping = torch.arange(batch_size * seq_len)
        context_lens = torch.tensor([seq_len, seq_len])
        
        output = attn(x, is_prefill=True, block_tables=block_tables,
                     slot_mapping=slot_mapping, context_lens=context_lens)
        assert output.shape == (batch_size, seq_len, 256)
        
        print("  ✓ Prefill 模式测试通过")
        
        # Decode 模式
        x_decode = torch.randn(batch_size, 1, 256)
        slot_mapping_decode = torch.tensor([32, 33])
        context_lens_decode = torch.tensor([32, 32])
        
        output = attn(x_decode, is_prefill=False, block_tables=block_tables,
                     slot_mapping=slot_mapping_decode, context_lens=context_lens_decode)
        assert output.shape == (batch_size, 1, 256)
        
        print("  ✓ Decode 模式测试通过")
        print("✅ PagedAttention 测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ PagedAttention 测试失败: {e}\n")
        return False


def test_transformer():
    """测试 Transformer 模型"""
    print("测试 2: Transformer 模型...")
    try:
        from vllm_transformer import PagedTransformerLM
        
        model = PagedTransformerLM(
            d_model=256,
            n_head=4,
            vocab_size=1000,
            max_seq_len=512,
            d_ff=1024,
            theta=10000.0,
            n_layer=2,
            num_kv_blocks=64,
            block_size=16,
            device='cpu'
        )
        
        batch_size = 2
        seq_len = 32
        x = torch.randint(0, 1000, (batch_size, seq_len))
        
        # 训练模式
        model.train()
        logits = model(x, is_prefill=True, block_tables=None)
        assert logits.shape == (batch_size, seq_len, 1000)
        print("  ✓ 训练模式测试通过")
        
        # 推理模式
        model.eval()
        with torch.no_grad():
            logits = model(x, is_prefill=True, block_tables=None)
        assert logits.shape == (batch_size, seq_len, 1000)
        print("  ✓ 推理模式测试通过")
        
        print("✅ Transformer 模型测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ Transformer 模型测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_sequence_manager():
    """测试序列管理"""
    print("测试 3: 序列管理...")
    try:
        from engine.sequence import Sequence, SamplingParams, SequenceStatus
        
        params = SamplingParams(temperature=0.8, max_tokens=50)
        seq = Sequence([1, 2, 3, 4], params)
        
        assert len(seq) == 4
        assert seq.num_prompt_tokens == 4
        assert seq.num_completion_tokens == 0
        assert seq.status == SequenceStatus.WAITING
        
        seq.append_token(5)
        assert len(seq) == 5
        assert seq.num_completion_tokens == 1
        
        print("  ✓ 序列操作测试通过")
        print("✅ 序列管理测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ 序列管理测试失败: {e}\n")
        return False


def test_block_manager():
    """测试块管理器"""
    print("测试 4: 块管理器...")
    try:
        from engine.block_manager import BlockManager
        from engine.sequence import Sequence, SamplingParams
        
        manager = BlockManager(num_blocks=64, block_size=16)
        
        # 创建序列
        seq1 = Sequence([1, 2, 3] * 10, SamplingParams())  # 30 tokens
        seq2 = Sequence([1, 2, 3] * 5, SamplingParams())   # 15 tokens
        
        # 分配块
        assert manager.can_allocate(seq1)
        manager.allocate(seq1)
        assert len(seq1.block_table) == 2  # 需要 2 个块 (30/16 向上取整)
        
        assert manager.can_allocate(seq2)
        manager.allocate(seq2)
        assert len(seq2.block_table) == 1  # 需要 1 个块
        
        print("  ✓ 块分配测试通过")
        
        # 回收块
        manager.deallocate(seq1)
        assert len(seq1.block_table) == 0
        
        print("  ✓ 块回收测试通过")
        print("✅ 块管理器测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ 块管理器测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_scheduler():
    """测试调度器"""
    print("测试 5: 调度器...")
    try:
        from engine.scheduler import Scheduler
        from engine.sequence import Sequence, SamplingParams
        
        scheduler = Scheduler(
            num_kv_blocks=64,
            block_size=16,
            max_num_seqs=4,
            max_num_batched_tokens=128
        )
        
        # 添加请求
        seq1 = Sequence([1, 2, 3] * 5, SamplingParams(max_tokens=10))
        seq2 = Sequence([4, 5, 6] * 5, SamplingParams(max_tokens=10))
        
        scheduler.add(seq1)
        scheduler.add(seq2)
        
        # 第一次调度（Prefill）
        seqs, is_prefill = scheduler.schedule()
        assert is_prefill == True
        assert len(seqs) == 2
        
        print("  ✓ Prefill 调度测试通过")
        
        # 模拟生成 token
        scheduler.postprocess(seqs, [10, 11])
        
        # 第二次调度（Decode）
        seqs, is_prefill = scheduler.schedule()
        assert is_prefill == False
        assert len(seqs) == 2
        
        print("  ✓ Decode 调度测试通过")
        print("✅ 调度器测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ 调度器测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_engine():
    """测试推理引擎"""
    print("测试 6: 推理引擎...")
    try:
        from engine.llm_engine import LLMEngine, SamplingParams
        from vllm_transformer import PagedTransformerLM
        
        model = PagedTransformerLM(
            d_model=128,
            n_head=4,
            vocab_size=1000,
            max_seq_len=256,
            d_ff=512,
            theta=10000.0,
            n_layer=2,
            num_kv_blocks=64,
            block_size=16,
            device='cpu'
        )
        
        engine = LLMEngine(
            model=model,
            num_kv_blocks=64,
            block_size=16,
            device='cpu'
        )
        
        # 添加请求
        engine.add_request([1, 2, 3], SamplingParams(max_tokens=5))
        engine.add_request([4, 5, 6], SamplingParams(max_tokens=5))
        
        # 执行几步
        for i in range(10):
            if engine.is_finished():
                break
            outputs, num_tokens = engine.step()
            print(f"  Step {i+1}: {'Prefill' if num_tokens > 0 else 'Decode'}, {abs(num_tokens)} tokens")
        
        print("✅ 推理引擎测试通过\n")
        return True
        
    except Exception as e:
        print(f"❌ 推理引擎测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始运行测试套件")
    print("=" * 60 + "\n")
    
    results = []
    
    results.append(("PagedAttention", test_attention()))
    results.append(("Transformer", test_transformer()))
    results.append(("序列管理", test_sequence_manager()))
    results.append(("块管理器", test_block_manager()))
    results.append(("调度器", test_scheduler()))
    results.append(("推理引擎", test_engine()))
    
    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name:20s}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！系统可以正常使用。")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)