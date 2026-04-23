from kfp import compiler
from pipeline.pipeline import pipeline

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.yaml",
    )