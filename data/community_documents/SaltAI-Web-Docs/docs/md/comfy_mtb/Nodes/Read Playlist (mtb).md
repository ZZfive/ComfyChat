# Read Playlist (mtb)
## Documentation
- Class name: `Read Playlist (mtb)`
- Category: `mtb/IO`
- Output node: `False`

The Read Playlist node is designed to facilitate the reading of playlists from a specified path, allowing for the conditional loading of playlist data based on user-defined parameters. It abstracts the complexities of file path resolution and conditional logic, providing a streamlined approach to accessing and utilizing playlist information.
## Input types
### Required
- **`enable`**
    - Determines whether the reading of the playlist is enabled or not, affecting whether the operation proceeds.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`persistant_playlist`**
    - Indicates whether the playlist should be read from a persistent storage location, influencing the path resolution for the playlist.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`playlist_name`**
    - The name of the playlist to be read, which can include a format specifier for indexing, affecting the final resolved file name.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`index`**
    - An index that can be used within the playlist name format specifier, allowing for dynamic naming of playlists.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`playlist`**
    - Comfy dtype: `PLAYLIST`
    - The playlist data loaded from the specified file, if available.
    - Python dtype: `List[Optional[dict]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ReadPlaylist:
    """Read a playlist"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("BOOLEAN", {"default": True}),
                "persistant_playlist": ("BOOLEAN", {"default": False}),
                "playlist_name": ("STRING", {"default": "playlist_{index:04d}"}),
                "index": ("INT", {"default": 0, "min": 0}),
            }
        }

    RETURN_TYPES = ("PLAYLIST",)
    FUNCTION = "read_playlist"
    CATEGORY = "mtb/IO"

    def read_playlist(
        self, enable: bool, persistant_playlist: bool, playlist_name: str, index: int
    ):
        playlist_name = playlist_name.format(index=index)
        playlist_path = get_playlist_path(playlist_name, persistant_playlist)
        if not enable:
            return (None,)

        if not playlist_path.exists():
            log.warning(f"Playlist {playlist_path} does not exist, skipping")
            return (None,)

        log.debug(f"Reading playlist {playlist_path}")
        return (json.loads(playlist_path.read_text(encoding="utf-8")),)

```
