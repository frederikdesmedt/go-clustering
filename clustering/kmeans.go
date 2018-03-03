package clustering

import (
	"math/rand"
)

// Sampler samples a `k`th vector from a vector space with the vector having a maximum length of `max`.
type Sampler func(k int, max float64) Vector

func uniformSampler(creator VectorCreator) func(int, float64) Vector {
	return func(k int, max float64) Vector {
		uniformComponentGenerator := func(_ int) float64 {
			return rand.Float64()
		}
		// Uniformly generates a Vector in a unit cube and checks whether the resulting vector also fits in the unit sphere.
		// Decent for 3D, but watch out for high dimensions as `P(l > 1)` will increase.
		var vec Vector
		for vec = creator.Null(); vec.Length() == 0 || vec.Length() > 1; vec = creator.New(uniformComponentGenerator) {
		}
		return vec.MulScalar(max)
	}
}

// KMeans will perform K-Means clustering on this dataset with the initial centroids sampled from a uniform sampler.
func (dataset *Dataset) KMeans(k int) CentroidClusterer {
	if dataset.IsEmpty() {
		return []Vector{}
	}
	return dataset.KMeansWithSampler(k, uniformSampler(dataset.AsSlice()[0].Creator()))
}

// KMeansWithCentroids will perform K-Means clustering on this dataset with the initial centroids provided.
func (dataset *Dataset) KMeansWithCentroids(centroids ...Vector) CentroidClusterer {
	if dataset.IsEmpty() {
		return []Vector{}
	}
	for maxDelta := 1.0; maxDelta > 0.1; {
		buckets := collectClusters(dataset, centroids)
		deltas := createNewCentroids(&centroids, buckets)
		maxDelta = 0
		for _, delta := range deltas {
			if delta > maxDelta {
				maxDelta = delta
			}
		}
	}
	return centroids
}

// KMeansWithSampler will perform K-Means clustering on this dataset with the initial centroids sampled from the provided sampler.
func (dataset *Dataset) KMeansWithSampler(k int, sampler Sampler) CentroidClusterer {
	if dataset.IsEmpty() {
		return []Vector{}
	}
	centroids := makeCentroids(k, dataset, sampler)
	return dataset.KMeansWithCentroids(centroids...)
}

func makeCentroids(k int, dataset *Dataset, sampler Sampler) []Vector {
	centroids := make([]Vector, k)
	maxLen := dataset.Max().Length()
	for i := 0; i < k; i++ {
		centroids[i] = sampler(i, maxLen)
	}
	return centroids
}

func collectClusters(dataset *Dataset, centroids []Vector) []bucketCollector {
	k := len(centroids)
	buckets := make([]bucketCollector, k)
	for _, record := range dataset.AsSlice() {
		cluster := 0
		distToCluster := centroids[cluster].DistanceTo(record)
		for k, centroid := range centroids {
			distToCentroid := record.DistanceTo(centroid)
			if distToCentroid < distToCluster {
				cluster = k
				distToCluster = distToCentroid
			}
		}
		buckets[cluster].Collect(record)
	}
	return buckets
}

func createNewCentroids(centroids *[]Vector, buckets []bucketCollector) []float64 {
	k := len(*centroids)
	deltas := make([]float64, k)
	for i := 0; i < k; i++ {
		if newCentroid := buckets[i].Average(); newCentroid != nil {
			deltas[i] = (*centroids)[i].DistanceTo(newCentroid)
			(*centroids)[i] = newCentroid
		}
	}
	return deltas
}

type bucketCollector struct {
	average       Vector
	normalization int
}

func (collector *bucketCollector) Collect(vec Vector) {
	if collector == nil || collector.average == nil {
		collector.average = vec
	} else {
		collector.average = collector.average.Add(vec)
	}
	collector.normalization++
}

func (collector *bucketCollector) Average() Vector {
	if collector == nil || collector.average == nil {
		return nil
	}

	return collector.average.MulScalar(1 / float64(collector.normalization))
}
