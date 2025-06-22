import utils.job_generator as generator

if __name__ == "__main__":
    print("Generating Cluster:")
    generator.generate_cluster_state(debug=True)

    print("Generating Jobs:")
    generator.generate_pending_jobs(debug=True)