"""mdas/api — FastAPI wrapper around the mdas ingestion/inference/adapter pipeline.

Pure routing: every endpoint delegates straight into mdas.ingestion,
mdas.inference, or mdas.adapter and passes their pydantic models through as
request/response bodies. No epidemiological logic lives here.
"""
