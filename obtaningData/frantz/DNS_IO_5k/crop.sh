ls -1 *.png | cut -d . -f 1 | while read line ; do convert $line.png -crop 627x624+194+45 $line.mini.png; done
