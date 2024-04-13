ffmpeg -i input.mp4 out%05d.png


# heic convert

find . -iname "*.heic" -exec heif-convert -q 100 {} {}.png \;

magick convert image.heic image.jpg
convert *.heif -set filename:base "%[basename]" "%[filename:base].jpg"
mogrify -format png *.heif