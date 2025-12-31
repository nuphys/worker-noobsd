#!/usr/bin/env python3
"""
Test script for LoRA functionality in the worker.

This script tests:
1. Schema validation with and without LoRA parameters
2. LoRA cache path generation
3. Backward compatibility (requests without LoRA)
"""

import sys
from schemas import INPUT_SCHEMA
from runpod.serverless.utils.rp_validator import validate


def test_schema_validation():
    """Test that the input schema properly validates LoRA parameters."""
    print("=" * 70)
    print("TEST 1: Schema Validation")
    print("=" * 70)
    
    # Test 1a: Request without LoRA (backward compatibility)
    print("\n1a. Testing backward compatibility (no LoRA)...")
    test_input_no_lora = {
        'prompt': 'A beautiful landscape',
        'height': 1024,
        'width': 1024,
        'num_inference_steps': 28,
        'guidance_scale': 5.5,
        'seed': 42,
        'scheduler': 'K_EULER_ANCESTRAL',
        'num_images': 1
    }
    
    result = validate(test_input_no_lora, INPUT_SCHEMA)
    if 'validated_input' in result:
        loras = result['validated_input'].get('loras')
        print(f"   ✓ PASSED: Request validated, loras field = {loras}")
        assert loras is None, "LoRA field should be None when not specified"
    else:
        print(f"   ✗ FAILED: {result.get('errors')}")
        return False
    
    # Test 1b: Request with LoRA as string (simple format)
    print("\n1b. Testing LoRA with string format...")
    test_input_lora_string = {
        'prompt': 'A beautiful landscape',
        'loras': ['https://example.com/lora.safetensors']
    }
    
    result = validate(test_input_lora_string, INPUT_SCHEMA)
    if 'validated_input' in result:
        loras = result['validated_input'].get('loras')
        print(f"   ✓ PASSED: Request validated, loras = {loras}")
        assert loras == ['https://example.com/lora.safetensors'], "LoRA URL should be preserved"
    else:
        print(f"   ✗ FAILED: {result.get('errors')}")
        return False
    
    # Test 1c: Request with LoRA as dict (with scale)
    print("\n1c. Testing LoRA with dict format (custom scale)...")
    test_input_lora_dict = {
        'prompt': 'A beautiful landscape',
        'loras': [
            {'path': 'https://example.com/lora.safetensors', 'scale': 0.8}
        ]
    }
    
    result = validate(test_input_lora_dict, INPUT_SCHEMA)
    if 'validated_input' in result:
        loras = result['validated_input'].get('loras')
        print(f"   ✓ PASSED: Request validated, loras = {loras}")
        assert isinstance(loras, list), "LoRA should be a list"
        assert loras[0]['scale'] == 0.8, "LoRA scale should be preserved"
    else:
        print(f"   ✗ FAILED: {result.get('errors')}")
        return False
    
    # Test 1d: Request with multiple LoRAs
    print("\n1d. Testing multiple LoRAs...")
    test_input_multi_lora = {
        'prompt': 'A beautiful landscape',
        'loras': [
            {'path': 'https://example.com/lora1.safetensors', 'scale': 0.8},
            {'path': 'https://example.com/lora2.safetensors', 'scale': 1.2}
        ]
    }
    
    result = validate(test_input_multi_lora, INPUT_SCHEMA)
    if 'validated_input' in result:
        loras = result['validated_input'].get('loras')
        print(f"   ✓ PASSED: Request validated with {len(loras)} LoRAs")
        assert len(loras) == 2, "Should have 2 LoRAs"
    else:
        print(f"   ✗ FAILED: {result.get('errors')}")
        return False
    
    return True


def test_cache_path_generation():
    """Test that LoRA cache paths are generated correctly."""
    print("\n" + "=" * 70)
    print("TEST 2: Cache Path Generation")
    print("=" * 70)
    
    import os
    import hashlib
    import re
    
    LORA_CACHE_DIR = '/models/loras'
    
    def get_lora_cache_path(lora_source):
        """Simplified version for testing (without makedirs)."""
        if lora_source.startswith(LORA_CACHE_DIR):
            return lora_source
        
        if '/' not in lora_source and '\\' not in lora_source:
            return os.path.join(LORA_CACHE_DIR, lora_source)
        
        if lora_source.startswith('http://') or lora_source.startswith('https://'):
            url_parts = lora_source.split('?')[0]
            filename = url_parts.split('/')[-1]
            
            if not filename or '.' not in filename or filename in ['download', 'resolve']:
                url_hash = hashlib.md5(lora_source.encode()).hexdigest()[:12]
                filename = f'lora_{url_hash}.safetensors'
            
            filename = re.sub(r'[^\w\-.]', '_', filename)
        else:
            filename = lora_source.replace('/', '_').replace('\\', '_')
            if not filename.endswith(('.safetensors', '.pt', '.bin')):
                filename += '.safetensors'
        
        return os.path.join(LORA_CACHE_DIR, filename)
    
    tests = [
        ('https://example.com/my-lora.safetensors', 'my-lora.safetensors'),
        ('my-cached-lora.safetensors', 'my-cached-lora.safetensors'),
        ('https://civitai.com/api/download/models/123456', 'lora_'),
        ('username/repo-name', 'username_repo-name.safetensors'),
    ]
    
    all_passed = True
    for i, (test_input, expected_part) in enumerate(tests, 1):
        result = get_lora_cache_path(test_input)
        if expected_part in result and result.startswith(LORA_CACHE_DIR):
            print(f"\n2{chr(96+i)}. ✓ PASSED: {test_input[:50]}")
            print(f"    → {result}")
        else:
            print(f"\n2{chr(96+i)}. ✗ FAILED: {test_input}")
            print(f"    Expected to contain: {expected_part}")
            print(f"    Got: {result}")
            all_passed = False
    
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LoRA Functionality Test Suite")
    print("=" * 70)
    
    results = []
    
    # Run tests
    try:
        results.append(("Schema Validation", test_schema_validation()))
    except Exception as e:
        print(f"\n✗ Schema validation test failed with exception: {e}")
        results.append(("Schema Validation", False))
    
    try:
        results.append(("Cache Path Generation", test_cache_path_generation()))
    except Exception as e:
        print(f"\n✗ Cache path generation test failed with exception: {e}")
        results.append(("Cache Path Generation", False))
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
