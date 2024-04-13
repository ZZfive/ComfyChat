# Add To Playlist (mtb)
## Documentation
- Class name: `Add To Playlist (mtb)`
- Category: `mtb/IO`
- Output node: `True`

The node is designed to add videos to a specified playlist, with options for using relative paths and making the playlist persistent. It dynamically formats the playlist name based on an index and ensures the creation of necessary directories, handling both new and existing playlists.
## Input types
### Required
- **`relative_paths`**
    - Determines whether the paths of videos added to the playlist should be relative to the output directory. This affects how the video paths are stored and accessed within the playlist.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`persistant_playlist`**
    - Indicates if the playlist should be stored in a persistent manner, affecting its storage location and potentially its longevity.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`playlist_name`**
    - The name of the playlist, which can be dynamically formatted using the index. This name is used to identify and manage the playlist file.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`index`**
    - An integer used to format the playlist name dynamically, allowing for the creation of sequentially named playlists.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AddToPlaylist:
    """Add a video to the playlist"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "relative_paths": ("BOOLEAN", {"default": False}),
                "persistant_playlist": ("BOOLEAN", {"default": False}),
                "playlist_name": ("STRING", {"default": "playlist_{index:04d}"}),
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "add_to_playlist"
    CATEGORY = "mtb/IO"

    def add_to_playlist(
        self,
        relative_paths: bool,
        persistant_playlist: bool,
        playlist_name: str,
        index: int,
        **kwargs,
    ):
        playlist_name = playlist_name.format(index=index)
        playlist_path = get_playlist_path(playlist_name, persistant_playlist)

        if not playlist_path.parent.exists():
            playlist_path.parent.mkdir(parents=True, exist_ok=True)

        playlist = []
        if not playlist_path.exists():
            playlist_path.write_text("[]")
        else:
            playlist = json.loads(playlist_path.read_text())
        log.debug(f"Playlist {playlist_path} has {len(playlist)} items")
        for video in kwargs.values():
            if relative_paths:
                video = Path(video).relative_to(output_dir).as_posix()

            log.debug(f"Adding {video} to playlist")
            playlist.append(video)

        log.debug(f"Writing playlist {playlist_path}")
        playlist_path.write_text(json.dumps(playlist), encoding="utf-8")
        return ()

```
