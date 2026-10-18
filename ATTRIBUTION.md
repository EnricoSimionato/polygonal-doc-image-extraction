# Attribution

The training pages of PolyDETR are synthesised from the following openly licensed image collections. The full list,
one row per image with license, author, title and source URL, is in
[attribution/training-assets.csv](attribution/training-assets.csv) (19,444 images).

| Source | Images | License |
|---|---|---|
| [Open Images](https://storage.googleapis.com/openimages/web/index.html) (Flickr photos), including 500 cut-outs made from them | 13,291 | CC BY 2.0 (as listed by Open Images; Google makes no warranty on the license status of each image) |
| [Wikimedia Commons](https://commons.wikimedia.org) (photos, diagrams, line drawings, textures) | 4,929 | public domain, CC0, CC BY 2.0-4.0, CC BY-SA 2.0-4.0, and 29 images under GPL, GFDL or Free Art License |
| [Unsplash](https://unsplash.com) via [Lorem Picsum](https://picsum.photos), including 231 cut-outs | 1,224 | [Unsplash License](https://unsplash.com/license) |

Totals by license family: CC BY 2.0 68.7%, CC BY-SA 12.6%, public domain / CC0 / no restrictions 9.8%, Unsplash 6.3%,
other CC BY 2.4%, copyleft (GPL, GFDL, FAL) 0.1%, other 0.1%.

Model backbone initialisation: [`timm/resnet18.a1_in1k`](https://huggingface.co/timm/resnet18.a1_in1k) (Apache-2.0),
trained on ImageNet-1k.

Thanks to all the photographers and contributors who share their work under open licenses.
