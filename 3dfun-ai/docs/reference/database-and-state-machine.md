# Generations, events, and credits

The course build stores users, uploads, generations, assets, attempts, events, and credit transactions in SQLite. The app opens a SQLAlchemy session per request. Local files and previews live under the configured storage root. See `app/db.py` for the tables and constraints.

## Generation lifecycle

Creating a generation reserves credits and queues the job. During a normal job, the worker advances through validation, processing, geometry, texturing, optimization, preview rendering, and completion. Some providers skip individual stages. `app/state_machine.py` defines the allowed transitions. The API reports `COMPLETED`, `FAILED`, or `CANCELLED` at the end of an attempt. Eligible failed and cancelled jobs can be queued again.

The worker records events with increasing sequence numbers. The event endpoint replays stored events and streams later updates through the local event bus. The course configuration runs the queue and event bus in one process.

## Credits

`app/credits.py` manages reservation, capture, and refund against the user's wallet. Successful jobs capture the reserved credits. Cancelled or failed jobs release the reservation according to their state. The ledger records the corresponding transactions, which appear in the dashboard. See the [REST API reference](api-spec.md) for job and download routes.
