from graphrag.index import run_pipeline_with_config, create_pipeline_config
import os
import asyncio

pipeline_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "settings.yaml"
)

async def run():
    
    outputs = []
    async for output in run_pipeline_with_config(pipeline_file):
        outputs.append(output)

    pipeline_result = outputs[-1]
    
    if pipeline_result.result is not None:
        top_nodes = pipeline_result.result.head(10)
        print("pipeline result", top_nodes)
    else:
        print("No results!")
        
if __name__ == "__main__":
    asyncio.run(run())