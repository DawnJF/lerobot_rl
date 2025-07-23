import logging
import time

repo_id
fps = 10


def record_dataset():
    """
    Record a dataset of robot interactions using either a policy or teleop.

    This function runs episodes in the environment and records the observations,
    actions, and results for dataset creation.

    Args:
        env: The environment to record from.
        policy: Optional policy to generate actions (if None, uses teleop).
        cfg: Configuration object containing recording parameters like:
            - repo_id: Repository ID for dataset storage
            - dataset_root: Local root directory for dataset
            - num_episodes: Number of episodes to record
            - fps: Frames per second for recording
            - push_to_hub: Whether to push dataset to Hugging Face Hub
            - task: Name/description of the task being recorded
            - number_of_steps_after_success: Number of additional steps to continue recording after
                                  a success (reward=1) is detected. This helps collect
                                  more positive examples for reward classifier training.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    action_names = ["delta_x_ee", "delta_y_ee", "delta_z_ee"]
    action_names.append("gripper_delta")

    # Configure dataset features based on environment spaces
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": env.observation_space["observation.state"].shape,
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "complementary_info.discrete_penalty": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["discrete_penalty"],
        },
    }

    # Add image features
    for key in env.observation_space:
        if "image" in key:
            features[key] = {
                "dtype": "video",
                "shape": env.observation_space[key].shape,
                "names": ["channels", "height", "width"],
            }

    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id,
        cfg.fps,
        root=cfg.dataset_root,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=features,
    )

    # Record episodes
    episode_index = 0
    recorded_action = None
    while episode_index < cfg.num_episodes:
        obs, _ = env.reset()
        start_episode_t = time.perf_counter()
        log_say(f"Recording episode {episode_index}", play_sounds=True)

        # Track success state collection
        success_detected = False
        success_steps_collected = 0

        # Run episode steps
        while time.perf_counter() - start_episode_t < cfg.wrapper.control_time_s:
            start_loop_t = time.perf_counter()

            # Get action from policy if available
            if cfg.pretrained_policy_name_or_path is not None:
                action = policy.select_action(obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Check if episode needs to be rerecorded
            if info.get("rerecord_episode", False):
                break

            # For teleop, get action from intervention
            recorded_action = {
                "action": (
                    info["action_intervention"].cpu().squeeze(0).float()
                    if policy is None
                    else action
                )
            }

            # Process observation for dataset
            obs_processed = {k: v.cpu().squeeze(0).float() for k, v in obs.items()}

            # Check if we've just detected success
            if reward == 1.0 and not success_detected:
                success_detected = True
                logging.info("Success detected! Collecting additional success states.")

            # Add frame to dataset - continue marking as success even during extra collection steps
            frame = {**obs_processed, **recorded_action}

            # If we're in the success collection phase, keep marking rewards as 1.0
            if success_detected:
                frame["next.reward"] = np.array([1.0], dtype=np.float32)
            else:
                frame["next.reward"] = np.array([reward], dtype=np.float32)

            # Only mark as done if we're truly done (reached end or collected enough success states)
            really_done = terminated or truncated
            if success_detected:
                success_steps_collected += 1
                really_done = (
                    success_steps_collected >= cfg.number_of_steps_after_success
                )

            frame["next.done"] = np.array([really_done], dtype=bool)
            frame["complementary_info.discrete_penalty"] = torch.tensor(
                [info.get("discrete_penalty", 0.0)], dtype=torch.float32
            )
            dataset.add_frame(frame, task=cfg.task)

            # Maintain consistent timing
            if cfg.fps:
                dt_s = time.perf_counter() - start_loop_t
                busy_wait(1 / cfg.fps - dt_s)

            # Check if we should end the episode
            if (terminated or truncated) and not success_detected:
                # Regular termination without success
                break
            elif (
                success_detected
                and success_steps_collected >= cfg.number_of_steps_after_success
            ):
                # We've collected enough success states
                logging.info(
                    f"Collected {success_steps_collected} additional success states"
                )
                break

        # Handle episode recording
        if info.get("rerecord_episode", False):
            dataset.clear_episode_buffer()
            logging.info(f"Re-recording episode {episode_index}")
            continue

        dataset.save_episode()
        episode_index += 1

    # Finalize dataset
    # dataset.consolidate(run_compute_stats=True)
    if cfg.push_to_hub:
        dataset.push_to_hub()
