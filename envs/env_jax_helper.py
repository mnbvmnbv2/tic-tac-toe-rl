"""Rollout wrapper for gymnax environments."""

import functools
import jax
import jax.numpy as jnp
import gymnax


class RolloutWrapper(object):
    """Wrapper to define batch evaluation for generation parameters."""

    def __init__(
        self,
        env: gymnax.environments.environment.Environment,
        env_params: gymnax.environments.environment.EnvParams,
        # num_env_steps: int,
    ):
        """Wrapper to define batch evaluation for generation parameters."""
        # Define the RL environment & network forward function
        self.env = env
        self.env_params = env_params
        # self.num_env_steps = num_env_steps

    @functools.partial(jax.jit, static_argnums=(0,))
    def population_rollout(self, rng_eval, policy_params):
        """Reshape parameter vector and evaluate the generation."""
        # Evaluate population of nets on gymnax task - vmap over rng & params
        pop_rollout = jax.vmap(self.batch_rollout, in_axes=(None, 0))
        return pop_rollout(rng_eval, policy_params)

    @functools.partial(jax.jit, static_argnums=(0,))
    def batch_rollout(self, rng_eval, policy_params):
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0, None))
        return batch_rollout(rng_eval, policy_params)

    @functools.partial(jax.jit, static_argnums=(0,))
    def single_rollout(self, rng_input, policy_params):
        """Rollout a pendulum episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, policy_params, rng, cum_reward, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            # if self.model_forward is not None:
            #     action = self.model_forward(policy_params, obs, rng_net)
            # else:
            action = self.env.action_space(self.env_params).sample(rng_net)
            next_obs, next_state, reward, done, _ = self.env.step(
                rng_step, state, action, self.env_params
            )
            new_cum_reward = cum_reward + reward * valid_mask
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_obs,
                next_state,
                policy_params,
                rng,
                new_cum_reward,
                new_valid_mask,
            ]
            y = [obs, action, reward, next_obs, done]
            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                policy_params,
                rng_episode,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            self.env_params.max_steps_in_episode,
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        obs, action, reward, next_obs, done = scan_out
        cum_return = carry_out[-2]
        return obs, action, reward, next_obs, done, cum_return

    @property
    def input_shape(self):
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, _ = self.env.reset(rng, self.env_params)
        return obs.shape
