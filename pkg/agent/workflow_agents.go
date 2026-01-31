package agent

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// SequentialAgent executes a list of agents in order, passing accumulated
// state between them. The last agent's output becomes the final result.
type SequentialAgent struct {
	name   string
	agents []Agent
}

// NewSequentialAgent creates a new SequentialAgent that runs agents in order.
func NewSequentialAgent(name string, agents []Agent) *SequentialAgent {
	return &SequentialAgent{name: name, agents: agents}
}

func (a *SequentialAgent) Name() string {
	return a.name
}

func (a *SequentialAgent) SubAgents() []Agent {
	return a.agents
}

func (a *SequentialAgent) Execute(ctx context.Context, task *Task) (*Result, error) {
	result := &Result{
		TaskID:  task.ID,
		Success: false,
		Steps:   []ExecutionStep{},
	}

	for _, ag := range a.agents {
		stepStart := time.Now()

		subResult, err := ag.Execute(ctx, task)

		// Record step
		step := ExecutionStep{
			AgentName: ag.Name(),
			Action:    "execute",
			Duration:  time.Since(stepStart),
			Timestamp: stepStart,
		}

		if err != nil {
			step.Error = err.Error()
			result.Steps = append(result.Steps, step)
			result.Error = fmt.Sprintf("agent %s failed: %v", ag.Name(), err)
			return result, err
		}

		// Merge steps and state
		result.Steps = append(result.Steps, subResult.Steps...)

		// Last agent's output is final
		result.Output = subResult.Output
		result.Artifacts = append(result.Artifacts, subResult.Artifacts...)
	}

	result.Success = true
	return result, nil
}

// ParallelAgent executes agents concurrently. Each agent receives its own
// copy of the task state. Results are merged after all agents complete.
type ParallelAgent struct {
	name   string
	agents []Agent
}

// NewParallelAgent creates a new ParallelAgent that runs agents concurrently.
func NewParallelAgent(name string, agents []Agent) *ParallelAgent {
	return &ParallelAgent{name: name, agents: agents}
}

func (a *ParallelAgent) Name() string {
	return a.name
}

func (a *ParallelAgent) SubAgents() []Agent {
	return a.agents
}

func (a *ParallelAgent) Execute(ctx context.Context, task *Task) (*Result, error) {
	result := &Result{
		TaskID:  task.ID,
		Success: false,
		Steps:   []ExecutionStep{},
	}

	type agentResult struct {
		result *Result
		err    error
	}

	results := make([]agentResult, len(a.agents))
	var wg sync.WaitGroup

	for i, ag := range a.agents {
		wg.Add(1)
		go func(idx int, ag Agent) {
			defer wg.Done()

			// Each agent gets its own state copy
			taskCopy := *task
			taskCopy.State = make(map[string]interface{})
			for k, v := range task.State {
				taskCopy.State[k] = v
			}

			res, err := ag.Execute(ctx, &taskCopy)
			results[idx] = agentResult{result: res, err: err}
		}(i, ag)
	}

	wg.Wait()

	// Merge results
	outputs := make(map[string]interface{})

	for i, res := range results {
		if res.err != nil {
			result.Error = fmt.Sprintf("agent %s failed: %v", a.agents[i].Name(), res.err)
			return result, res.err
		}

		result.Steps = append(result.Steps, res.result.Steps...)
		result.Artifacts = append(result.Artifacts, res.result.Artifacts...)

		// Collect outputs by agent name
		outputs[a.agents[i].Name()] = res.result.Output

		// Merge state deltas back
		if len(res.result.Steps) > 0 {
			lastStep := res.result.Steps[len(res.result.Steps)-1]
			for k, v := range lastStep.StateDelta {
				task.State[k] = v
			}
		}
	}

	result.Output = outputs
	result.Success = true
	return result, nil
}

// PipelineAgent chains agents where each stage's output becomes the next
// stage's input, forming a data processing pipeline.
type PipelineAgent struct {
	name   string
	stages []Agent
}

// NewPipelineAgent creates a new PipelineAgent that chains agents sequentially
// with data flow between stages.
func NewPipelineAgent(name string, stages []Agent) *PipelineAgent {
	return &PipelineAgent{name: name, stages: stages}
}

func (a *PipelineAgent) Name() string {
	return a.name
}

func (a *PipelineAgent) SubAgents() []Agent {
	return a.stages
}

func (a *PipelineAgent) Execute(ctx context.Context, task *Task) (*Result, error) {
	result := &Result{
		TaskID:  task.ID,
		Success: false,
		Steps:   []ExecutionStep{},
	}

	// Each stage receives previous stage's output as input
	currentInput := task.Input

	for _, stage := range a.stages {
		// Update task input from previous output
		task.Input = currentInput

		subResult, err := stage.Execute(ctx, task)
		if err != nil {
			result.Error = fmt.Sprintf("stage %s failed: %v", stage.Name(), err)
			return result, err
		}

		result.Steps = append(result.Steps, subResult.Steps...)
		result.Artifacts = append(result.Artifacts, subResult.Artifacts...)

		// Output becomes input for next stage
		if str, ok := subResult.Output.(string); ok {
			currentInput = str
		} else {
			currentInput = fmt.Sprint(subResult.Output)
		}
	}

	result.Output = currentInput
	result.Success = true
	return result, nil
}
