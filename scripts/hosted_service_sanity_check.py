#!/usr/bin/env python3
"""
Sanity check script for the hosted service.
Validates that configuration files and code match the expected state for deployment.
"""

import tomli
import sys
import importlib.util
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Path to the morphik-core directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_toml_configuration():
    """Verify morphik.toml has the expected configuration for hosted service"""
    try:
        with open(os.path.join(BASE_DIR, "morphik.toml"), "rb") as f:
            config = tomli.load(f)
        
        # Critical checks - these must match exactly
        expected_values = {
            ("api", "host"): "127.0.0.1",
            ("auth", "dev_mode"): False,
            ("completion", "model"): "openai_gpt4o",
            ("embedding", "model"): "openai_embedding",
            ("embedding", "dimensions"): 1536,
            ("embedding", "similarity_metric"): "dotProduct",
            ("parser", "contextual_chunking_model"): "openai_gpt4o",
            ("parser.vision", "model"): "openai_gpt4o",
            ("storage", "provider"): "aws-s3",
            ("storage", "bucket_name"): "morphik-storage",
            ("rules", "model"): "openai_gpt4o",
            ("morphik", "mode"): "cloud",
            ("graph", "model"): "openai_gpt4o",
            ("telemetry", "service_name"): "databridge-core-hosted",
        }
        
        errors = []
        for (section, key), expected_value in expected_values.items():
            if "." in section:
                # Handle nested sections like parser.vision
                parent, child = section.split(".")
                if parent not in config or child not in config[parent]:
                    errors.append(f"Missing [{parent}] section or {child} key")
                    continue
                actual_value = config[parent][child].get(key)
            else:
                if section not in config:
                    errors.append(f"Missing [{section}] section")
                    continue
                actual_value = config[section].get(key)
            
            if actual_value != expected_value:
                errors.append(f"[{section}] {key} expected '{expected_value}', found '{actual_value}'")
        
        if errors:
            logger.error("❌ morphik.toml configuration errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("✅ morphik.toml configuration verified")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error checking morphik.toml: {str(e)}")
        return False

def check_colpali_embedding_model():
    """Verify the colpali_embedding_model.py has the expected imports and model name"""
    try:
        file_path = os.path.join(BASE_DIR, "core", "embedding", "colpali_embedding_model.py")
        
        # Check the file content directly instead of importing the module
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check imports
        import_errors = []
        
        # Check if the correct imports are present
        if "from colpali_engine.models import ColIdefics3, ColIdefics3Processor" not in content:
            import_errors.append("Missing 'from colpali_engine.models import ColIdefics3, ColIdefics3Processor'")
        
        # The file should contain model_name = "vidore/colSmol-256M"
        if 'model_name = "vidore/colSmol-256M"' not in content:
            import_errors.append("Expected model_name = \"vidore/colSmol-256M\" not found")
        
        # Check model class
        if "class ColpaliEmbeddingModel" not in content:
            import_errors.append("ColpaliEmbeddingModel class not found")
            
        # Check attn_implementation
        if 'attn_implementation="eager"' not in content:
            import_errors.append("Expected attn_implementation=\"eager\" not found")
            
        # Check if model initialization is correct
        if "self.model = ColIdefics3.from_pretrained" not in content:
            import_errors.append("Missing ColIdefics3.from_pretrained initialization")
            
        # Check processor initialization
        if "self.processor = ColIdefics3Processor.from_pretrained" not in content:
            import_errors.append("Missing ColIdefics3Processor.from_pretrained initialization")
        
        if import_errors:
            logger.error("❌ colpali_embedding_model.py errors:")
            for error in import_errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("✅ colpali_embedding_model.py verified")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error checking colpali_embedding_model.py: {str(e)}")
        return False

def check_start_server():
    """Verify start_server.py has the expected configuration"""
    try:
        file_path = os.path.join(BASE_DIR, "start_server.py")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        errors = []
        
        # Check that Redis startup is commented out
        if "# check_and_start_redis()" not in content:
            errors.append("Redis startup should be commented out (check_and_start_redis)")
        
        # Check for skip_ollama_check override
        if "args.skip_ollama_check = True" not in content:
            errors.append("Missing 'args.skip_ollama_check = True' override")
        
        if errors:
            logger.error("❌ start_server.py errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        logger.info("✅ start_server.py verified")
        return True
    
    except Exception as e:
        logger.error(f"❌ Error checking start_server.py: {str(e)}")
        return False

def run_all_checks():
    """Run all sanity checks and report status"""
    logger.info("Running hosted service sanity checks...")
    
    checks = [
        ("morphik.toml", check_toml_configuration),
        ("colpali_embedding_model.py", check_colpali_embedding_model),
        ("start_server.py", check_start_server),
    ]
    
    all_passed = True
    for name, check_func in checks:
        logger.info(f"Checking {name}...")
        passed = check_func()
        all_passed = all_passed and passed
    
    if all_passed:
        logger.info("✅ All checks passed! Configuration is ready for hosted service deployment.")
        return 0
    else:
        logger.error("❌ Some checks failed. Please fix the issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_checks())
