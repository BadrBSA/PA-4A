from google.cloud import aiplatform


import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "core-site-423419-g8-6f60b4d05c41.json"

huggingface_token = "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"
os.environ["HUGGING_FACE_HUB_TOKEN"] = huggingface_token

# Initialize Vertex AI
aiplatform.init(project='core-site-423419-g8', location='europe-west1')


SERVING_DOCKER_URI = "us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve"

model_id = "mistralai/Mistral-7B-v0.1_version2"

machine_type = "g2-standard-8"
accelerator_type = "NVIDIA_L4"
accelerator_count = 1

# Fill with the created service account.
service_account = "intellisum@core-site-423419-g8.iam.gserviceaccount.com"

endpoint = aiplatform.Endpoint.create(display_name=f"{model_id}-endpoint")


vllm_args = [
    "--host=0.0.0.0",
    "--port=7080",
    f"--model={model_id}",
    f"--tensor-parallel-size={accelerator_count}",
    "--swap-space=16",
    "--gpu-memory-utilization=0.9",
    "--disable-log-stats",
]

model = aiplatform.Model.upload(
    display_name=model_id,
    artifact_uri="gs://data-intellisum/models/model_v1",
    serving_container_image_uri=SERVING_DOCKER_URI,
    serving_container_command=["python", "-m", "vllm.entrypoints.api_server"],
    serving_container_args=vllm_args,
    serving_container_ports=[7080],
    serving_container_predict_route="/generate",
    serving_container_health_route="/ping",
)


model.deploy(
    endpoint=endpoint,
    machine_type=machine_type,
    accelerator_type=accelerator_type,
    accelerator_count=accelerator_count,
    deploy_request_timeout=1800,
    service_account=service_account,
)