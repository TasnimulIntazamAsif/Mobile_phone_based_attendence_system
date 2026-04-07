# Mobile Phone Based Attendance System (API Only)

This project is now API-only (no UI flow).  
It provides face enrollment and attendance verification APIs using base64 images, with Swagger documentation.

## Features

- Enroll employee face images from base64
- Train/update face encodings after enrollment
- Verify attendance from base64 image
- Match/unmatch response handling
- Attendance tracking with:
  - `employee_id`
  - `name`
  - `date`
  - `arrival`
  - `exit`
  - `status`
- Swagger/OpenAPI docs

## Project Structure (Important Data Files)

- `data/dataset/` -> stored training images (`<employee_id>_<name>/`)
- `data/encodings.pkl` -> trained face encodings
- `data/attendance.csv` -> attendance records
- `data/base64_registry.json` -> enrollment base64 info registry

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Server runs at:

- `http://127.0.0.1:5000`

## Swagger / OpenAPI

- Swagger UI: `GET /api/docs`
- OpenAPI JSON: `GET /api/openapi.json`

## Main APIs

### 1) Enroll Face

`POST /api/v1/enroll-face`

Request:

```json
{
  "employee_id": "570508",
  "employee_name": "Asif",
  "base64_images": [
    "data:image/jpeg;base64,...",
    "data:image/jpeg;base64,..."
  ]
}
```

Response (success):

```json
{
  "ok": true,
  "message": "Enrollment completed and face model refreshed",
  "employee_id": "570508",
  "employee_name": "Asif",
  "accepted_images": 2,
  "trained": true
}
```

### 2) Verify Attendance

`POST /api/v1/verify-attendance`

Request:

```json
{
  "image_base64": "data:image/jpeg;base64,..."
}
```

Response (matched):

```json
{
  "ok": true,
  "matched": true,
  "message": "Arrival Marked (13:22:08)",
  "employee_id": "570508",
  "name": "Asif",
  "date": "2026-04-07",
  "arrival": "13:22:08",
  "exit": "",
  "status": "Arrived"
}
```

Response (unmatched):

```json
{
  "ok": true,
  "matched": false,
  "message": "Unmatched"
}
```

### 3) Inspect Stored Data

`GET /api/stored-data`

Returns attendance rows, registered users, and base64 registry summary.

## Notes

- Use clear face images for best results.
- Enrollment should include multiple angles per employee.
- If matching quality drops, re-enroll employee images and retrain through `enroll-face`.
# Mobile_phone_based_attendence_system