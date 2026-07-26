from inference_sdk import InferenceHTTPClient
import base64
client = InferenceHTTPClient(
    api_url="http://127.0.0.1:9001",
    api_key="anQm43Si6DlwZMvBM0LC",
)

result = client.run_workflow(
    workspace_name="juless-workspace-zidwe",
    workflow_id="fire-and-smoke-segmentation-alerts-1784557902550",
    images={
        "image": "../Downloads/fires/images.jpeg"
    },
)
print(result)


img_b64 = result[0]["output_image"] if isinstance(result, list) else result["output_image"]

with open("fire_smoke_output.jpg", "wb") as f:
    f.write(base64.b64decode(img_b64))