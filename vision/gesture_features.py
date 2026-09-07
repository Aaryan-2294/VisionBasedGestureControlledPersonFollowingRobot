import numpy as np


def extract_features(hand):
    points = np.array(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand],
        dtype=np.float32
    )

    wrist = points[0]
    points = points - wrist

    scale = np.linalg.norm(points[9])
    if scale < 1e-8:
        return np.zeros(63, dtype=np.float32).tolist()

    points = points / scale

    index_mcp = points[5]
    middle_mcp = points[9]

    x_axis = index_mcp
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)

    y_axis = middle_mcp
    y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

    normalized = np.column_stack([
        points @ x_axis,
        points @ y_axis,
        points @ z_axis
    ])

    return normalized.flatten().tolist()