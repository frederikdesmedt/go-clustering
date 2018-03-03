package main

import (
	"math/rand"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"github.com/frederikdesmedt/clustering"
	"github.com/lucasb-eyer/go-colorful"
)

func main() {
	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "Dataset coloured according to clusters"

	data := generateData(200)

	clusterer := data.KMeansWithCentroids(
		clustering.Vector2d(0, 0),
		clustering.Vector2d(1, 0),
		clustering.Vector2d(0, 1),
		clustering.Vector2d(1, 1),
	)
	// clusterer := data.KMeans(4)
	clusterColouring := make(map[clustering.Cluster]colorful.Color)
	clusters := clusterer.Clusters()
	if colours, err := colorful.HappyPalette(len(clusters)); err != nil {
		panic(err)
	} else {
		for i := 0; i < len(colours); i++ {
			clusterColouring[clusters[i]] = colours[i]
		}
	}

	if partitions, err := clusterer.ClusteredPartition(&data); err != nil {
		panic(err)
	} else {
		for cluster, partition := range partitions {
			if partitionScatter, err := plotter.NewScatter(asXYs(partition)); err != nil {
				panic(err)
			} else {
				partitionScatter.GlyphStyle.Color = clusterColouring[cluster]
				partitionScatter.GlyphStyle.Shape = draw.CircleGlyph{}
				p.Add(partitionScatter)
			}
		}
	}

	for cluster, centroid := range clusterer.Centroids() {
		if centroidScatter, err := plotter.NewScatter(asXYs([]clustering.Vector{centroid})); err != nil {
			panic(err)
		} else {
			centroidScatter.GlyphStyle.Shape = draw.PyramidGlyph{}
			centroidScatter.GlyphStyle.Color = clusterColouring[cluster]
			centroidScatter.GlyphStyle.Radius = 0.15 * vg.Centimeter
			p.Add(centroidScatter)
		}
	}

	if err := p.Save(20*vg.Centimeter, 20*vg.Centimeter, "kmeans.png"); err != nil {
		panic(err)
	}
}

func asXYs(vecs []clustering.Vector) plotter.XYs {
	xys := make(plotter.XYs, len(vecs))
	for i, vec := range vecs {
		if vec2, ok := vec.(clustering.Vector2); !ok {
			panic("Provided vector is not a Vector2")
		} else {
			xys[i].X = vec2[0]
			xys[i].Y = vec2[1]
		}
	}
	return xys
}

// randomPoints returns some random x, y points.
func generateData(n int) clustering.Dataset {
	dataset := make([]clustering.Vector, n, n)
	for i := 0; i < n; i++ {
		x := rand.Float64()
		y := rand.Float64()
		dataset[i] = clustering.Vector2d(x, y)
	}
	return clustering.CreateNonEmptyDataset(dataset)
}
