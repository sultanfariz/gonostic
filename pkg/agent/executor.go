package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

// Executor manages async task execution with a pool of workers.
type Executor struct {
	agent       Agent
	jobs        map[string]*Job
	mu          sync.RWMutex
	workerCount int
	jobQueue    chan *Job
}

// Job represents a submitted task and its execution state.
type Job struct {
	Task   *Task
	Result *Result
	Status JobStatus
	Error  error
}

// JobStatus represents the lifecycle state of a job.
type JobStatus string

const (
	JobPending   JobStatus = "pending"
	JobRunning   JobStatus = "running"
	JobCompleted JobStatus = "completed"
	JobFailed    JobStatus = "failed"
)

// NewExecutor creates a new Executor with the given agent and worker pool size.
func NewExecutor(agent Agent, workerCount int) *Executor {
	if workerCount == 0 {
		workerCount = 5
	}

	ex := &Executor{
		agent:       agent,
		jobs:        make(map[string]*Job),
		workerCount: workerCount,
		jobQueue:    make(chan *Job, 100),
	}

	// Start workers
	for i := 0; i < workerCount; i++ {
		go ex.worker()
	}

	return ex
}

// Submit creates and queues a new job, returning the task ID for tracking.
func (e *Executor) Submit(input string, params map[string]interface{}, config *ExecutionConfig) (string, error) {
	taskID := uuid.New().String()

	task := &Task{
		ID:        taskID,
		Input:     input,
		Params:    params,
		State:     make(map[string]interface{}),
		Config:    config,
		StartedAt: time.Now(),
	}

	// Copy params to state
	for k, v := range params {
		task.State[k] = v
	}

	job := &Job{
		Task:   task,
		Status: JobPending,
	}

	e.mu.Lock()
	e.jobs[taskID] = job
	e.mu.Unlock()

	// Queue for execution
	e.jobQueue <- job

	return taskID, nil
}

// GetStatus returns the current status of a job.
func (e *Executor) GetStatus(taskID string) (JobStatus, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	job, ok := e.jobs[taskID]
	if !ok {
		return "", fmt.Errorf("task not found: %s", taskID)
	}

	return job.Status, nil
}

// GetResult returns the job result. It blocks until the job is complete.
func (e *Executor) GetResult(taskID string) (*Result, error) {
	e.mu.RLock()
	job, ok := e.jobs[taskID]
	e.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("task not found: %s", taskID)
	}

	// Poll until complete
	for {
		e.mu.RLock()
		status := job.Status
		e.mu.RUnlock()

		if status == JobCompleted {
			return job.Result, nil
		}
		if status == JobFailed {
			return job.Result, job.Error
		}

		time.Sleep(100 * time.Millisecond)
	}
}

// worker processes jobs from the queue.
func (e *Executor) worker() {
	for job := range e.jobQueue {
		e.executeJob(job)
	}
}

func (e *Executor) executeJob(job *Job) {
	// Update status
	e.mu.Lock()
	job.Status = JobRunning
	e.mu.Unlock()

	// Create context with timeout
	ctx := context.Background()
	if job.Task.Config != nil && job.Task.Config.TimeoutSeconds > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, time.Duration(job.Task.Config.TimeoutSeconds)*time.Second)
		defer cancel()
	}

	// Execute agent
	result, err := e.agent.Execute(ctx, job.Task)

	job.Task.CompletedAt = time.Now()
	job.Result = result
	job.Error = err

	// Update final status
	e.mu.Lock()
	if err != nil {
		job.Status = JobFailed
	} else {
		job.Status = JobCompleted
	}
	e.mu.Unlock()
}

// ExecuteSync executes a task synchronously and returns the result directly.
func (e *Executor) ExecuteSync(ctx context.Context, input string, params map[string]interface{}) (*Result, error) {
	task := &Task{
		ID:        uuid.New().String(),
		Input:     input,
		Params:    params,
		State:     make(map[string]interface{}),
		StartedAt: time.Now(),
	}

	for k, v := range params {
		task.State[k] = v
	}

	return e.agent.Execute(ctx, task)
}
