IH26M_RJOINTS_ORDER = (
    "Thumb_4",
    "Thumb_3",
    "Thumb_2",
    "Thumb_1",
    "Index_4",
    "Index_3",
    "Index_2",
    "Index_1",
    "Middle_4",
    "Middle_3",
    "Middle_2",
    "Middle_1",
    "Ring_4",
    "Ring_3",
    "Ring_2",
    "Ring_1",
    "Pinky_4",
    "Pinky_3",
    "Pinky_2",
    "Pinky_1",
    "Wrist",
)
HO3D_JOINTS_ORDER = (
    "Wrist",

    "Index_1",
    "Index_2",
    "Index_3",

    "Middle_1",
    "Middle_2",
    "Middle_3",

    "Pinky_1",
    "Pinky_2",
    "Pinky_3",

    "Ring_1",
    "Ring_2",
    "Ring_3",

    "Thumb_1",
    "Thumb_2",
    "Thumb_3",

    "Thumb_4",
    "Index_4",
    "Middle_4",
    "Ring_4",
    "Pinky_4",
)
MANO_JOINTS_ORDER = (
    "Wrist",
    "Index_1",
    "Index_2",
    "Index_3",
    "Middle_1",
    "Middle_2",
    "Middle_3",
    "Pinky_1",
    "Pinky_2",
    "Pinky_3",
    "Ring_1",
    "Ring_2",
    "Ring_3",
    "Thumb_1",
    "Thumb_2",
    "Thumb_3",
)

TARGET_JOINTS_ORDER = (
    "Wrist",
    "Thumb_1",
    "Thumb_2",
    "Thumb_3",
    "Thumb_4",
    "Index_1",
    "Index_2",
    "Index_3",
    "Index_4",
    "Middle_1",
    "Middle_2",
    "Middle_3",
    "Middle_4",
    "Ring_1",
    "Ring_2",
    "Ring_3",
    "Ring_4",
    "Pinky_1",
    "Pinky_2",
    "Pinky_3",
    "Pinky_4",
)

TARGET_JOINTS_CONNECTION = [
    (0, 1),   # Wrist -> Thumb_1
    (0, 5),   # Wrist -> Index_1
    (0, 9),   # Wrist -> Middle_1
    (0, 13),  # Wrist -> Ring_1
    (0, 17),  # Wrist -> Pinky_1
    # Thumb
    (1, 2),   # Thumb_1 -> Thumb_2
    (2, 3),   # Thumb_2 -> Thumb_3
    (3, 4),   # Thumb_3 -> Thumb_4
    # Index
    (5, 6),   # Index_1 -> Index_2
    (6, 7),   # Index_2 -> Index_3
    (7, 8),   # Index_3 -> Index_4
    # Middle
    (9, 10),  # Middle_1 -> Middle_2
    (10, 11), # Middle_2 -> Middle_3
    (11, 12), # Middle_3 -> Middle_4
    # Ring
    (13, 14), # Ring_1 -> Ring_2
    (14, 15), # Ring_2 -> Ring_3
    (15, 16), # Ring_3 -> Ring_4
    # Pinky
    (17, 18), # Pinky_1 -> Pinky_2
    (18, 19), # Pinky_2 -> Pinky_3
    (19, 20)  # Pinky_3 -> Pinky_4
]