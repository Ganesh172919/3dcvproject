# Architecture and data flow

## Photo to mesh

1. The web Studio sends the image to the FastAPI upload route. The backend checks the file size and image contents and records an upload.
2. Creating a generation reserves credits and queues the selected provider. The job emits stage updates that the Studio displays.
3. The Depth Anything V2 and MiDaS paths combine a foreground mask with relative depth. The engine creates a volume using the silhouette and a thickness prior, extracts and cleans the surface, projects the photo into a texture, and writes the selected export formats. The TripoSR path uses a learned density field to build its mesh.
4. The job stores the output files and creates an asset record. The Studio displays the result in a WebGL viewer, and asset endpoints handle downloads.

The local install uses a SQLite database at `data/3dfun.db` and file storage under `data/storage/` by default. The default in-process job runner belongs to the server process. Keep one worker for the reference setup.

## Training and evidence

TinyDepth is a separate command-line experiment. It renders labeled views from source meshes, fits a depth model, exports a verified ONNX graph, and projects the visible surface into a point cloud using a foreground mask and known camera intrinsics. It does not run on Studio startup. The website shows the pipeline and can copy a command for use in the terminal.

The course benchmark compares exported meshes. TinyDepth measures predicted camera-axis depth and can compare partial clouds. Different inputs, references and coordinate conventions keep those results separate.

**Evidence** `app/routers/uploads.py` · `app/jobs.py` · `engine/recon/pipeline.py` · `app/providers.py` · `training/pipeline.py`
