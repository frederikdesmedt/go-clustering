package clustering

import (
	"math/rand"
	"reflect"
)

// VectorCreator is able to create a vector of some real abstract vector space.
// A VectorCreator is a separate interface because an arbitrary vector must be able to be created without having an initial vector.
type VectorCreator interface {
	// New should create a new vector of some real abstract vector space with the components set as specified by the provided function.
	// Note how the function might not be completely used, e.g., if the new vector only has 3 components only `f(0)`, `f(1)`, and `f(2)` are used.
	New(func(int) float64) Vector
	// Null creates a null-vector in this real abstract vector space.
	Null() Vector
}

// Vector represents a vector in an abstract real vector space.
type Vector interface {
	// Add adds two vectors by component-wise addition and returns the result.
	Add(Vector) Vector
	// Subtract subtracts the other vector from this vector, i.e., `v - other`.
	Subtract(Vector) Vector
	// MulScalar multiplies this vector with a scalar.
	MulScalar(float64) Vector
	// TransposedMul multiplies the transpose of this vector with the other vector.
	TransposedMul(Vector) float64
	// Length calculates the length of this vector.
	Length() float64
	// Normalize will calculate the vector in the same direction but with a length of 1. When this vector is the null-vector a random vector with length 1 is returned.
	Normalize() Vector
	// DistanceTo will return the distance between this vector and the other vector.
	DistanceTo(Vector) float64
	// Creator will return a VectorCreator creating vectors of this kind.
	Creator() VectorCreator
}

// Vector2 is a real vector with 2 components.
type Vector2 [2]float64

// Dataset is an indexed list of vectors representing some kind of dataset.
type Dataset struct {
	data    []Vector
	creator VectorCreator
}

// CreateDataset will create a dataset containing the provided data.
func CreateDataset(data []Vector, creator VectorCreator) Dataset {
	return Dataset{data: data, creator: creator}
}

// CreateNonEmptyDataset will create a dataset containing the provided non-empty slice of data.
func CreateNonEmptyDataset(data []Vector) Dataset {
	if len(data) == 0 {
		panic("Expected a non-empty dataset, but got an empty dataset")
	}
	return CreateDataset(data, data[0].Creator())
}

type vector2Creator struct{}

func (v vector2Creator) New(f func(int) float64) Vector {
	return Vector2{f(0), f(1)}
}

func (v vector2Creator) Null() Vector {
	return Vector2{0, 0}
}

func checkVector2(v Vector) Vector2 {
	v2, ok := v.(Vector2)
	if !ok {
		panic("Expected a Vector2 but got " + reflect.TypeOf(v).Name())
	}
	return v2
}

// Add adds two vectors by component-wise addition and returns the result.
func (v Vector2) Add(other Vector) Vector {
	otherv := checkVector2(other)
	return Vector2{
		v[0] + otherv[0],
		v[1] + otherv[1],
	}
}

// Subtract subtracts the other vector from this vector, i.e., `v - other`.
func (v Vector2) Subtract(other Vector) Vector {
	otherv := checkVector2(other)
	return Vector2{
		v[0] - otherv[0],
		v[1] - otherv[1],
	}
}

// MulScalar multiplies this vector with a scalar.
func (v Vector2) MulScalar(other float64) Vector {
	return Vector2{
		v[0] * other,
		v[1] * other,
	}
}

// TransposedMul multiplies the transpose of this vector with the other vector.
func (v Vector2) TransposedMul(other Vector) float64 {
	otherv := checkVector2(other)
	return v[0]*otherv[0] + v[1]*otherv[1]
}

// Length calculates the length of this vector.
func (v Vector2) Length() float64 {
	return v.TransposedMul(v)
}

// Normalize will calculate the vector in the same direction but with a length of 1. When this vector is the null-vector a random vector with length 1 is returned.
func (v Vector2) Normalize() Vector {
	if v.Length() == 0 {
		return Vector2d(rand.Float64(), rand.Float64())
	}

	return v.MulScalar(1 / v.Length())
}

// DistanceTo will return the distance between this vector and the other vector.
func (v Vector2) DistanceTo(other Vector) float64 {
	otherv := checkVector2(other)
	return v.Subtract(otherv).Length()
}

// Creator will return a VectorCreator creating Vector2s.
func (v Vector2) Creator() VectorCreator {
	return vector2Creator{}
}

// Vector2d creates a new 2-dimensional vector with the supplied values as its components.
func Vector2d(x, y float64) Vector2 {
	return Vector2{x, y}
}

// Max will return the largest vector in the dataset, if there are multiple largest vectors, the first is returned, if the dataset is empty, a vector with size 0 is returned.
func (dataset *Dataset) Max() Vector {
	result := dataset.creator.Null()
	length := result.Length()
	for _, vec := range dataset.AsSlice() {
		vecLen := vec.Length()
		if vecLen > length {
			result = vec
			length = vecLen
		}
	}
	return result
}

// Count will return the number of data points in this dataset.
func (dataset *Dataset) Count() int {
	return len(dataset.data)
}

// IsEmpty returns true if and only if this dataset is empty.
func (dataset *Dataset) IsEmpty() bool {
	return dataset.Count() == 0
}

// AsSlice will give back a slice with the elements of this dataset in the same order.
func (dataset *Dataset) AsSlice() []Vector {
	return dataset.data
}
