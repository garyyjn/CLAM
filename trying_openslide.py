import openslide

slide = openslide.OpenSlide("/Users/M261759/Documents/GitHub/CLAM/slides/TCGA-49-4505-11A-01-BS1.e019788b-29ba-4469-bda7-c39824386c12.svs")

print(slide.dimensions)