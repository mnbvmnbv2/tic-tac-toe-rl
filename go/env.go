package main

import (
	"fmt"
	"math/rand"
	"time"
)

type Env struct {
	gs   [18]uint8
	w, r uint8
}

func (e *Env) reset()     { e.gs = [18]uint8{}; e.w, e.r = 0, 0 }
func (e *Env) step(a int) {}

func main() {
	const B = 1000
	envs := make([]Env, B)
	for i := range envs {
		envs[i].reset()
	}
	start := time.Now()
	steps := 0
	for time.Since(start) < time.Second {
		for i := range envs {
			envs[i].step(rand.Intn(9))
		}
		steps += B
	}
	fmt.Printf("Go: %d steps/s\n", steps)
}
