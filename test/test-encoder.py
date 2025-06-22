import utils.job_generator as generator
import utils.state_encoder as encoder

if __name__ == "__main__":

    cluster = generator.generate_cluster_state()
    jobs = generator.generate_pending_jobs()
    print(encoder.encode_state(jobs,cluster))