import time

from garage.logger import logger, tabular
from garage.experiment.local_tf_runner import LocalRunner


class LocalMamlRunner(LocalRunner):

    def setup(self, algo, envs, sampler_cls=None, sampler_args=None):

        self.algo = algo
        self.envs = envs
        self.policy = self.algo.policy

        if sampler_args is None:
            sampler_args = {}

        if sampler_cls is None:
            from garage.tf.samplers import MultiEnvVectorizedSampler
            sampler_cls = MultiEnvVectorizedSampler

        self.sampler = sampler_cls(algo, envs, **sampler_args)

        self.initialize_tf_vars()
        self.has_setup = True

    def train(self,
              n_epochs,
              batch_size,
              n_epoch_cycles=1,
              plot=False,
              store_paths=False,
              pause_for_plot=False):
        """Start training.

        Args:
            n_epochs: Number of epochs.
            n_epoch_cycles: Number of batches of samples in each epoch.
                This is only useful for off-policy algorithm.
                For on-policy algorithm this value should always be 1.
            batch_size: Number of steps in batch.
            plot: Visualize policy by doing rollout after each epoch.
            store_paths: Save paths in snapshot.
            pause_for_plot: Pause for plot.

        Returns:
            The average return in last epoch cycle.

        """
        assert self.has_setup, "Use Runner.setup() to setup runner " \
                               "before training."

        self.n_epoch_cycles = n_epoch_cycles
        self.plot = plot
        self.start_worker()
        self.start_time = time.time()

        itr = 0
        last_return = None
        for epoch in range(n_epochs):
            self.itr_start_time = time.time()
            paths = None
            with logger.prefix('epoch #%d | ' % epoch):
                for cycle in range(n_epoch_cycles):
                    paths = self.obtain_samples(itr, batch_size)
                    paths = self.sampler.process_samples(itr, paths)
                    adaptation_data = [self.algo.policy_adapt_opt_values(p) for p in paths]

                    # Run another round of sampling here
                    paths = self.obtain_samples(itr, batch_size, adaptation_data)
                    paths = self.sampler.process_samples(itr, paths)
                    last_return = self.algo.train_once(itr, paths, adaptation_data)
                    itr += 1
                self.save_snapshot(epoch, paths if store_paths else None)
                self.log_diagnostics(pause_for_plot)

        self.shutdown_worker()
        return last_return

    def obtain_samples(self, itr, batch_size, adaptation_data=None):
        """Obtain one batch of samples.

        Args:
            itr: Index of iteration (epoch).
            batch_size: Number of steps in batch.
                This is a hint that the sampler may or may not respect.

        Returns:
            One batch of samples.

        """
        if adaptation_data:
            return self.sampler.obtain_samples(itr, batch_size, adaptation_data)
        else:
            return self.sampler.obtain_samples(itr, batch_size)

    def adapt(self, batch_size=4000, n_itr=1, env=None):
        if env is not None:
            from garage.tf.samplers import OnPolicyVectorizedSampler
            self.sampler = OnPolicyVectorizedSampler(self.algo, env, n_envs=2)

        self.start_worker()
        policy_params = self.policy.get_params()
        values_before_adapt = self.sess.run(policy_params)

        for itr in range(n_itr):
            # with logger.prefix('itr #%d | ' % itr):
            # logger.log('Obtaining samples...')
            paths = self.obtain_samples(itr, batch_size)
            # logger.log('Processing samples...')
            samples_data = self.sampler.process_samples(itr, paths)
            values = self.algo.policy_adapt_opt_values(samples_data)
            # logger.log('Computing adapted policy parameters...')
            params = self.algo.f_adapt(*values)
            self.policy.update_params(params)

        # Revert the policy as not adapted
        self.policy.update_params(values_before_adapt)

        return params