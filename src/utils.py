def get_split_by_artist(artist_name, train, val, test):
    if artist_name in train:
        return 'train'
    elif artist_name in val:
        return 'val'
    elif artist_name in test:
        return 'test'
