CLEANED_TRACKS = [
    # Track good
    ("2023-04-12-16-02-01", slice(None, 39037), 'train'),
    ("2023-04-12-16-02-01", slice(39038, None), 'val'),

    # Track good, no movement in the beginning
    ("2023-04-13-16-50-11", slice(7200, 65171), 'train'),
    ("2023-04-13-16-50-11", slice(65172, None), 'val'),

    # Track good
    ("2023-04-19-15-22-36", slice(None, 30221), 'train'),
    ("2023-04-19-15-22-36", slice(30222, None), 'val'),

    # Track good, no velocity in the end
    ("2023-04-20-17-33-33", slice(None, 42114), 'train'),
    ("2023-04-20-17-33-33", slice(42115, 45600), 'val'),

    # Track good, no velocity in the end
    ("2023-04-27-16-42-40", slice(None, 51836), 'train'),
    ("2023-04-27-16-42-40", slice(51837, 55500), 'val'),

    # Problem in the beginning, no movement in the middle
    ("2023-05-03-19-07-25", slice(5000, 18500), 'train'),
    ("2023-05-03-19-07-25", slice(20000, 24900), 'train'),
    ("2023-05-03-19-07-25", slice(25416, None), 'val'),

    # Data problems in the beginning
    ("2023-05-04-15-58-50", slice(9216, 36736), 'train'),
    ("2023-05-04-15-58-50", slice(36737, None), 'val'),

    # Some data problems in the middle
    ("2023-05-10-15-41-04", slice(None, 13000), 'train'),
    ("2023-05-10-15-41-04", slice(14000, None), 'val'),

    # Track good
    ("2023-05-11-17-08-21", slice(None, None), 'train'),

    # Problems with velocity
    # ("2023-05-11-17-31-38", slice(None, None), 'train'),

    # ("2023-05-11-17-54-42", slice(None, None), 'train'),
    # ("2023-05-11-18-21-37", slice(None, None), 'train'),
    # ("2023-05-11-18-39-23", slice(None, None), 'train'),

    ("2023-05-17-15-30-02", slice(None, 5800), 'train'),
    # Problems with data
    # ("2023-05-17-15-30-02", slice(5264, None), 'val'),

    # No movement in the end
    ("2023-05-18-16-40-47", slice(None, 9000), 'train'),
    ("2023-05-18-16-40-47", slice(9001, 10800), 'val'),

    # Track good
    ("2023-05-18-16-57-00", slice(None, 28767), 'train'),
    ("2023-05-18-16-57-00", slice(28768, None), 'val'),

    # Some data problems in the beginning and end
    ("2023-05-23-15-40-24", slice(3000, 47000), 'train'),
    ("2023-05-23-15-40-24", slice(47001, 52000), 'val'),

    # Data problems in the beginning
    ("2023-05-25-16-33-18", slice(6062, 42753), 'train'),
    ("2023-05-25-16-33-18", slice(42754, None), 'val'),

    # A lot of stopping, so a lot of filtered rows, but otherwise seems ok
    ("2023-05-30-15-42-35", slice(None, 50068), 'train'),
    ("2023-05-30-15-42-35", slice(50069, None), 'val'),

    # Good
    ("2023-06-01-18-10-55", slice(None, 33514), 'train'),
    ("2023-06-01-18-10-55", slice(33515, None), 'val'),

    # Some data problems
    ("2023-06-06-15-41-21", slice(None, 8200), 'train'),
    ("2023-06-06-15-41-21", slice(8300, 40000), 'train'),
    ("2023-06-06-15-41-21", slice(43000, None), 'val'),

    # Data problems in the beginning and end
    ("2023-06-08-18-50-17", slice(4500, 18200), 'train'),

    # Data problem in the beginning and end
    ("2023-06-08-19-18-03", slice(500, 2800), 'val'),

    ("2023-06-13-15-14-21", slice(7000, 15000), 'train'),
    ("2023-06-13-15-14-21", slice(15100, 24000), 'train'),
    ("2023-06-13-15-14-21", slice(24001, None), 'val'),

    # Some data problems
    ("2023-06-13-15-49-17", slice(None, 5200), 'train'),
    ("2023-06-13-15-49-17", slice(5500, 8700), 'train'),
    ("2023-06-13-15-49-17", slice(8800, 39000), 'train'),
    ("2023-06-13-15-49-17", slice(39001, 43000), 'val'),

    # Lots of data problems
    # ("2023-06-15-18-10-18", slice(None, 22029), 'train'),
    # ("2023-06-15-18-10-18", slice(22030, None), 'val'),

    # Good
    ("2023-06-30-12-11-33", slice(None, 61479), 'train'),
    ("2023-06-30-12-11-33", slice(61480, None), 'val'),

    # Good
    ("2023-07-04-15-04-53", slice(None, 20240), 'train'),
    ("2023-07-04-15-04-53", slice(20241, None), 'val'),

    # Jumps in position
    ("2023-07-06-12-20-35", slice(None, 5587), 'train'),
    ("2023-07-06-12-20-35", slice(5588, 7400), 'train'),
    ("2023-07-06-12-20-35", slice(7500, 8987), 'train'),
    ("2023-07-06-12-20-35", slice(12500, 24724), 'train'),
    ("2023-07-06-12-20-35", slice(24725, None), 'val'),

    # Looks good
    ("2023-07-07-13-26-44", slice(None, 57422), 'train'),
    ("2023-07-07-13-26-44", slice(57423, None), 'val'),

    # Looks good
    ("2023-07-11-15-44-44", slice(None, 60897), 'train'),
    ("2023-07-11-15-44-44", slice(60898, None), 'val'),

    # Looks good
    ("2023-07-13-10-42-27", slice(None, 66477), 'train'),
    ("2023-07-13-10-42-27", slice(66478, None), 'val'),

    ("2023-07-17-13-37-10", slice(None, 8500), 'train'),
    ("2023-07-17-13-37-10", slice(8700, 11800), 'train'),
    ("2023-07-17-13-37-10", slice(12000, 13900), 'train'),
    ("2023-07-17-13-37-10", slice(14100, 21600), 'train'),
    ("2023-07-17-13-37-10", slice(22500, 25000), 'train'),
    ("2023-07-17-13-37-10", slice(26000, 36000),  'train'),
    ("2023-07-17-13-37-10", slice(37000, 40000),  'train'),
    ("2023-07-17-13-37-10", slice(48000, 54500),  'val'),
    ("2023-07-17-14-38-28", slice(None, 19500),  'train'),
    ("2023-07-19-13-12-11", slice(3500, None),  'train'),
    # ("2023-07-21-11-52-18", slice(15000, None),  'train'),
    ("2023-07-24-13-53-29", slice(4500, 19000),  'train'),
    ("2023-07-24-14-25-47", slice(250, 1250),  'train'),
    ("2023-07-24-14-29-06", slice(4500, 16500), 'train'),
    ("2023-07-24-14-29-06", slice(18500, None), 'val'),
    # ("2023-07-26-14-22-18", slice(10000, None), 'train'),
    ("2023-07-27-14-58-24", slice(10000, None), 'train'),
    ("2023-07-27-15-46-09", slice(None, None),  'train'),
    ("2023-07-27-16-12-51", slice(None, None),  'train'),
    ("2023-08-01-15-47-18", slice(None, None),  'train'),
    # ("2023-08-02-16-27-51", slice(None, None), 'train'),
    # ("2023-08-04-11-15-22", slice(25000, None), 'train'),
    ("2023-08-08-15-40-29", slice(None, None),  'train'),
    ("2023-08-08-16-37-28", slice(None, None),  'train'),
    ("2023-08-09-13-44-25", slice(1500, 20000), 'train'),
    ("2023-08-09-13-44-25", slice(21000, 50500), 'train'),
    ("2023-08-09-13-44-25", slice(51000, None), 'val'),
    ("2023-08-09-14-07-47", slice(None, None),  'train'),
    ("2023-08-10-16-19-31", slice(None, None),  'train'),
    ("2023-08-17-16-16-29", slice(None, None),  'train'),
    # ("2023-08-21-15-17-51", slice(None, None),  'train'),
    # ("2023-08-22-15-25-30", slice(None, None),  'train'),
    ("2023-08-23-15-04-12", slice(None, None),  'train'),
    ("2023-08-23-15-12-38", slice(None, None),  'train'),
    ("2023-08-23-15-17-22", slice(None, None),  'train'),
    ("2023-08-23-15-21-21", slice(None, None),  'train'),
    ("2023-08-23-15-26-38", slice(None, None),  'train'),
    ("2023-08-23-15-57-55", slice(None, None),  'train'),
    ("2023-08-24-16-09-18", slice(12500, 43000),  'train'),
    ("2023-08-25-15-48-47", slice(None, None),  'train')
]