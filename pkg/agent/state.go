package agent

import "sync"

// MapState is a thread-safe implementation of the State interface
// backed by a map.
type MapState struct {
	mu   sync.RWMutex
	data map[string]interface{}
}

// NewMapState creates a new empty MapState.
func NewMapState() *MapState {
	return &MapState{
		data: make(map[string]interface{}),
	}
}

// NewMapStateFrom creates a MapState pre-populated with the given data.
func NewMapStateFrom(initial map[string]interface{}) *MapState {
	data := make(map[string]interface{}, len(initial))
	for k, v := range initial {
		data[k] = v
	}
	return &MapState{data: data}
}

func (s *MapState) Get(key string) (interface{}, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	v, ok := s.data[key]
	return v, ok
}

func (s *MapState) Set(key string, value interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data[key] = value
}

func (s *MapState) Delete(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.data, key)
}

func (s *MapState) Keys() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	keys := make([]string, 0, len(s.data))
	for k := range s.data {
		keys = append(keys, k)
	}
	return keys
}

func (s *MapState) Merge(delta map[string]interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for k, v := range delta {
		s.data[k] = v
	}
}

// TaskState is the working state of a Task. The zero value is empty and ready
// to use, so Task.State is never nil.
type TaskState struct {
	data map[string]interface{}
}

// Get returns the value for key.
func (s TaskState) Get(key string) (interface{}, bool) {
	v, ok := s.data[key]
	return v, ok
}

// Set stores value under key, allocating the map on first use.
func (s *TaskState) Set(key string, value interface{}) {
	if s.data == nil {
		s.data = make(map[string]interface{})
	}
	s.data[key] = value
}

// Merge sets every key in delta. A nil or empty delta does nothing.
func (s *TaskState) Merge(delta map[string]interface{}) {
	for k, v := range delta {
		s.Set(k, v)
	}
}

// Clone returns an independent copy. A zero value clones to a zero value.
func (s TaskState) Clone() TaskState {
	var c TaskState
	c.Merge(s.data)
	return c
}

// Range calls fn for every entry, in no fixed order.
func (s TaskState) Range(fn func(key string, value interface{})) {
	for k, v := range s.data {
		fn(k, v)
	}
}
