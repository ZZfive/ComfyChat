# ComfyUI-seam-carving
Seam carving (image resize) for ComfyUI

Adds node "Image Resize (seam carving)"

https://en.wikipedia.org/wiki/Seam_carving  
![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Broadway_tower_edit.jpg/640px-Broadway_tower_edit.jpg)
![](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Broadway_tower_edit_Seam_Carving.png/475px-Broadway_tower_edit_Seam_Carving.png)

Uses https://github.com/li-plus/seam-carving

Also known as liquid rescaling, changes image size by adding or removing rows or columns of pixels with least effect on resulting image.

Can be very slow if adding or removing many pixels, resize with regular methods closer to target size first.
