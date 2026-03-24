## 🐛 Problem

In `Phase 3 — Image Preprocessing`, the `deskew()` function has two weaknesses when dealing with real-world scanned election documents:

### Problem 1 — Hard cap at 5° silently skips correction

```python
MAX_DESKEW_ANGLE = 5.0

if abs(angle) > MAX_DESKEW_ANGLE:
    return img  # skipped silently
```

Scans tilted at **6°–10°** are returned unchanged → OCR reads skewed text/digits → higher Levenshtein error.

### Problem 2 — Dark fold lines poison HoughLines angle estimation

`cv2.HoughLines()` detects **all** strong lines, including dark fold creases. A fold line creates a dominant angle that skews the `np.median(angles)` calculation:

```
True table tilt:  1.5°
Fold line angle:  45° (diagonal crease)
Median result:    ~23° → exceeds MAX_DESKEW_ANGLE → deskew skipped entirely
```

→ Page that could have been corrected is returned as-is.

### Problem 3 — Phase ordering: Phase 2 runs BEFORE Phase 3

Table detection (Phase 2) operates on the **raw, possibly tilted image**. This means:
- Intersection count is lower on a tilted image (lines don't align perfectly with kernels)
- Risk of **False Negative** — valid table page rejected by Phase 2 before deskew can help

## ✅ Proposed Fixes

### Fix 1 — Denoise before Canny to suppress fold lines

```python
def deskew(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    denoised = cv2.fastNlMeansDenoising(gray, h=10)  # ✅ suppress fold noise
    edges = cv2.Canny(denoised, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
    ...
```

### Fix 2 — Expand MAX_DESKEW_ANGLE to 10°

```python
MAX_DESKEW_ANGLE = 10.0  # up from 5.0
```

### Fix 3 (Best for tables) — Projection Profile deskew

Rotate the image across a range of angles and pick the angle where horizontal pixel projection variance is maximized — this is the angle where table rows are most horizontal.

```python
def deskew_by_projection(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0
    best_variance = 0

    for angle in np.arange(-10.0, 10.0, 0.5):
        M = cv2.getRotationMatrix2D(
            (binary.shape[1] // 2, binary.shape[0] // 2), angle, 1.0
        )
        rotated = cv2.warpAffine(binary, M, (binary.shape[1], binary.shape[0]))
        projection = np.sum(rotated, axis=1)
        variance = np.var(projection)

        if variance > best_variance:
            best_variance = variance
            best_angle = angle

    M = cv2.getRotationMatrix2D(
        (img.shape[1] // 2, img.shape[0] // 2), best_angle, 1.0
    )
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
```

## 📊 Comparison

| Method | Tilt < 5° | Tilt 5–10° | Fold disruption | Speed |
|---|---|---|---|---|
| HoughLines current | ✅ | ❌ skipped | ⚠️ disrupted | 🚀 Fast |
| HoughLines + Denoise | ✅ | ❌ skipped | ✅ Better | 🚀 Fast |
| HoughLines + Denoise + 10° cap | ✅ | ✅ | ✅ Better | 🚀 Fast |
| Projection Profile | ✅ | ✅ | ✅ Best | 🐢 Slower |

**Recommended minimum fix:** Fix 1 + Fix 2 (low effort, high impact)  
**Recommended full fix:** Projection Profile for table pages

## 📁 File to Change

`src/phase3_preprocess/preprocess.py` — function `deskew()`

## 🔗 Related

- master-plan.md Phase 3: https://github.com/lumduan/superai-se6-mini-hackathon-2/blob/main/docs/plan/master-plan.md#phase-3--image-preprocessing
- Related issue: Phase 2 fold line false positive