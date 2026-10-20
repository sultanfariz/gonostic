package agent

import (
	"reflect"
	"testing"
)

// contents copies a TaskState into a plain map so tests compare full values.
func contents(s TaskState) map[string]interface{} {
	out := map[string]interface{}{}
	s.Range(func(k string, v interface{}) { out[k] = v })
	return out
}

func TestTaskState_Get(t *testing.T) {
	tests := []struct {
		name      string
		state     TaskState
		key       string
		wantValue interface{}
		wantOK    bool
	}{
		{"Success_zero_value_missing_key", TaskState{}, "k", nil, false},
		{"Success_existing_key", TaskState{data: map[string]interface{}{"k": "v"}}, "k", "v", true},
		{"Success_missing_key", TaskState{data: map[string]interface{}{"k": "v"}}, "other", nil, false},
		{"Success_nil_value_present", TaskState{data: map[string]interface{}{"k": nil}}, "k", nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := tt.state.Get(tt.key)
			if !reflect.DeepEqual(got, tt.wantValue) || ok != tt.wantOK {
				t.Errorf("Get(%q) = (%v, %v), want (%v, %v)", tt.key, got, ok, tt.wantValue, tt.wantOK)
			}
		})
	}
}

func TestTaskState_Set(t *testing.T) {
	tests := []struct {
		name  string
		state TaskState
		key   string
		value interface{}
		want  map[string]interface{}
	}{
		{"Success_zero_value_allocates", TaskState{}, "k", "v", map[string]interface{}{"k": "v"}},
		{"Success_adds_key", TaskState{data: map[string]interface{}{"a": 1}}, "b", 2, map[string]interface{}{"a": 1, "b": 2}},
		{"Success_overwrites_key", TaskState{data: map[string]interface{}{"a": 1}}, "a", 9, map[string]interface{}{"a": 9}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.state.Set(tt.key, tt.value)
			if got := contents(tt.state); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("state = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTaskState_Merge(t *testing.T) {
	tests := []struct {
		name  string
		state TaskState
		delta map[string]interface{}
		want  map[string]interface{}
	}{
		{"Success_nil_delta_does_nothing", TaskState{data: map[string]interface{}{"a": 1}}, nil, map[string]interface{}{"a": 1}},
		{"Success_empty_delta_on_zero_value", TaskState{}, map[string]interface{}{}, map[string]interface{}{}},
		{"Success_zero_value_gets_keys", TaskState{}, map[string]interface{}{"a": 1}, map[string]interface{}{"a": 1}},
		{"Success_overwrites_existing_key", TaskState{data: map[string]interface{}{"a": 1, "b": 2}}, map[string]interface{}{"a": 9, "c": 3}, map[string]interface{}{"a": 9, "b": 2, "c": 3}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.state.Merge(tt.delta)
			if got := contents(tt.state); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("state = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTaskState_Clone(t *testing.T) {
	tests := []struct {
		name string
		src  TaskState
		// mutate changes the clone; wantSrc and wantClone are the expected contents afterwards.
		mutate    func(c *TaskState, s *TaskState)
		wantSrc   map[string]interface{}
		wantClone map[string]interface{}
	}{
		{
			name:      "Success_clone_change_leaves_original",
			src:       TaskState{data: map[string]interface{}{"a": 1}},
			mutate:    func(c, s *TaskState) { c.Set("a", 2); c.Set("b", 3) },
			wantSrc:   map[string]interface{}{"a": 1},
			wantClone: map[string]interface{}{"a": 2, "b": 3},
		},
		{
			name:      "Success_original_change_leaves_clone",
			src:       TaskState{data: map[string]interface{}{"a": 1}},
			mutate:    func(c, s *TaskState) { s.Set("a", 2); s.Set("b", 3) },
			wantSrc:   map[string]interface{}{"a": 2, "b": 3},
			wantClone: map[string]interface{}{"a": 1},
		},
		{
			name:      "Success_zero_value_clone_is_usable",
			src:       TaskState{},
			mutate:    func(c, s *TaskState) { c.Set("a", 1) },
			wantSrc:   map[string]interface{}{},
			wantClone: map[string]interface{}{"a": 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := tt.src
			clone := src.Clone()
			tt.mutate(&clone, &src)
			if got := contents(src); !reflect.DeepEqual(got, tt.wantSrc) {
				t.Errorf("original = %v, want %v", got, tt.wantSrc)
			}
			if got := contents(clone); !reflect.DeepEqual(got, tt.wantClone) {
				t.Errorf("clone = %v, want %v", got, tt.wantClone)
			}
		})
	}
}

func TestTaskState_Range(t *testing.T) {
	tests := []struct {
		name  string
		state TaskState
		want  map[string]interface{}
	}{
		{"Success_zero_value_never_calls_fn", TaskState{}, map[string]interface{}{}},
		{"Success_visits_all_entries", TaskState{data: map[string]interface{}{"a": 1, "b": "x"}}, map[string]interface{}{"a": 1, "b": "x"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := map[string]interface{}{}
			calls := 0
			tt.state.Range(func(k string, v interface{}) {
				calls++
				got[k] = v
			})
			if calls != len(tt.want) {
				t.Errorf("fn calls = %d, want %d", calls, len(tt.want))
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("visited = %v, want %v", got, tt.want)
			}
		})
	}
}
