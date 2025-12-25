"""Video recording utilities for visualizing agent learning."""

from pathlib import Path
from typing import Callable, List, Optional, Union
import numpy as np

try:
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

from ..agents.goal_state_agent import GoalStateAgent, AgentConfig


def _check_gym():
    if not HAS_GYM:
        raise ImportError(
            "gymnasium is required for video recording. "
            "Install with: pip install gymnasium[classic-control]"
        )


def record_episode(
    agent: GoalStateAgent,
    env_name: str = "CartPole-v1",
    video_folder: str = "videos",
    name_prefix: str = "episode",
    max_steps: int = 500,
) -> int:
    """Record a single episode of the agent.

    Args:
        agent: Trained GoalStateAgent.
        env_name: Gymnasium environment name.
        video_folder: Directory to save videos.
        name_prefix: Prefix for video filename.
        max_steps: Maximum steps per episode.

    Returns:
        Number of steps in the episode.
    """
    _check_gym()

    Path(video_folder).mkdir(parents=True, exist_ok=True)

    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=name_prefix,
        episode_trigger=lambda x: True,
    )

    obs, info = env.reset()
    agent.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = agent.get_action(obs, training=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

    env.close()
    return steps


def record_learning_progression(
    config: Optional["AgentConfig"] = None,
    env_name: str = "CartPole-v1",
    max_episodes: int = 100,
    record_episodes: Optional[List[int]] = None,
    video_folder: str = "videos/learning_progression",
    max_steps_per_episode: int = 500,
    verbose: bool = True,
) -> dict:
    """Train an agent and record episodes at intervals to show learning.

    This function trains a new agent from scratch and records video of
    selected episodes to visualize the learning progression.

    Args:
        config: Agent configuration. Uses defaults if None.
        env_name: Gymnasium environment name.
        max_episodes: Maximum training episodes.
        record_episodes: List of episode numbers to record. If None,
            records at exponentially spaced intervals.
        video_folder: Directory to save videos.
        max_steps_per_episode: Maximum steps per episode.
        verbose: Whether to print progress.

    Returns:
        Dictionary with training metrics and recorded episode info.
    """
    _check_gym()

    from ..training import Trainer, TrainingConfig

    # Default recording schedule: exponential spacing
    if record_episodes is None:
        record_episodes = [0, 1, 2, 5, 10, 20, 50]
        record_episodes = [e for e in record_episodes if e < max_episodes]
        # Add final episode
        if max_episodes - 1 not in record_episodes:
            record_episodes.append(max_episodes - 1)

    record_episodes = sorted(set(record_episodes))

    Path(video_folder).mkdir(parents=True, exist_ok=True)

    # Create training config
    train_config = TrainingConfig(
        env_name=env_name,
        max_episodes=max_episodes,
        max_steps_per_episode=max_steps_per_episode,
        log_interval=10 if verbose else 1000,
        save_interval=0,  # Don't save checkpoints
    )

    if config:
        # Override with provided agent config
        train_config.action_learning_rate = config.action_learning_rate
        train_config.prediction_learning_rate = config.prediction_learning_rate
        train_config.action_update_steps = config.action_update_steps
        train_config.prediction_update_steps = config.prediction_update_steps
        train_config.use_adaptive_lr = config.use_adaptive_lr
        train_config.goal_indices = list(config.goal_indices)
        train_config.goal_values = list(config.goal_values)

    # Create trainer
    trainer = Trainer(config=train_config)

    # Training metrics
    episode_lengths = []
    goal_errors = []
    prediction_errors = []
    recorded_episodes = []

    if verbose:
        print(f"Training for {max_episodes} episodes")
        print(f"Recording episodes: {record_episodes}")
        print("-" * 50)

    for episode in range(max_episodes):
        # Check if we should record this episode
        if episode in record_episodes:
            # Record episode with current agent state
            video_path = Path(video_folder) / f"episode_{episode:04d}"
            video_path.mkdir(parents=True, exist_ok=True)

            # Create recording environment
            rec_env = gym.make(env_name, render_mode="rgb_array")
            rec_env = RecordVideo(
                rec_env,
                video_folder=str(video_path),
                name_prefix=f"ep{episode:04d}",
                episode_trigger=lambda x: True,
            )

            obs, _ = rec_env.reset()
            trainer.agent.reset()
            done = False
            rec_steps = 0

            while not done and rec_steps < max_steps_per_episode:
                action = trainer.agent.get_action(obs, training=False)
                obs, _, terminated, truncated, _ = rec_env.step(action)
                done = terminated or truncated
                rec_steps += 1

            rec_env.close()
            recorded_episodes.append({
                "episode": episode,
                "steps": rec_steps,
                "path": str(video_path),
            })

            if verbose:
                print(f"  Recorded episode {episode}: {rec_steps} steps")

        # Run training episode
        steps, goal_err, pred_err = trainer.run_episode()
        episode_lengths.append(steps)
        goal_errors.append(goal_err)
        prediction_errors.append(pred_err)

        if verbose and episode % train_config.log_interval == 0:
            avg_steps = np.mean(episode_lengths[-10:]) if episode_lengths else 0
            print(f"Episode {episode}: steps={steps}, avg={avg_steps:.1f}")

        # Check convergence
        if trainer._check_convergence():
            if verbose:
                print(f"Converged at episode {episode}")
            break

    # Record final state if not already recorded
    final_ep = len(episode_lengths) - 1
    if final_ep not in record_episodes and final_ep > 0:
        video_path = Path(video_folder) / f"episode_{final_ep:04d}_final"
        video_path.mkdir(parents=True, exist_ok=True)

        rec_env = gym.make(env_name, render_mode="rgb_array")
        rec_env = RecordVideo(
            rec_env,
            video_folder=str(video_path),
            name_prefix=f"ep{final_ep:04d}_final",
            episode_trigger=lambda x: True,
        )

        obs, _ = rec_env.reset()
        trainer.agent.reset()
        done = False
        rec_steps = 0

        while not done and rec_steps < max_steps_per_episode:
            action = trainer.agent.get_action(obs, training=False)
            obs, _, terminated, truncated, _ = rec_env.step(action)
            done = terminated or truncated
            rec_steps += 1

        rec_env.close()
        recorded_episodes.append({
            "episode": final_ep,
            "steps": rec_steps,
            "path": str(video_path),
            "final": True,
        })

        if verbose:
            print(f"  Recorded final episode {final_ep}: {rec_steps} steps")

    if verbose:
        print("-" * 50)
        print(f"Training complete. Videos saved to: {video_folder}")

    return {
        "episode_lengths": episode_lengths,
        "goal_errors": goal_errors,
        "prediction_errors": prediction_errors,
        "recorded_episodes": recorded_episodes,
        "agent": trainer.agent,
        "converged": trainer.converged,
        "convergence_episode": trainer.convergence_episode,
    }


def create_learning_montage(
    video_folder: str = "videos/learning_progression",
    output_path: str = "videos/learning_montage.mp4",
    fps: int = 30,
) -> Optional[str]:
    """Create a montage video from recorded learning progression.

    Requires moviepy to be installed: pip install moviepy

    Args:
        video_folder: Directory containing recorded episode videos.
        output_path: Path for output montage video.
        fps: Frames per second for output video.

    Returns:
        Path to created montage, or None if creation failed.
    """
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
    except ImportError:
        print("moviepy is required for creating montage. Install with: pip install moviepy")
        return None

    video_folder = Path(video_folder)
    if not video_folder.exists():
        print(f"Video folder not found: {video_folder}")
        return None

    # Find all mp4 files
    video_files = sorted(video_folder.rglob("*.mp4"))
    if not video_files:
        print(f"No video files found in {video_folder}")
        return None

    clips = []
    for vf in video_files:
        try:
            clip = VideoFileClip(str(vf))

            # Extract episode number from path
            ep_num = vf.parent.name.split("_")[1] if "_" in vf.parent.name else "?"

            # Add episode label
            txt = TextClip(
                f"Episode {ep_num}",
                fontsize=24,
                color='white',
                bg_color='black',
            ).set_position(('left', 'top')).set_duration(clip.duration)

            labeled_clip = CompositeVideoClip([clip, txt])
            clips.append(labeled_clip)
        except Exception as e:
            print(f"Warning: Could not load {vf}: {e}")

    if not clips:
        print("No valid video clips found")
        return None

    # Concatenate all clips
    final = concatenate_videoclips(clips, method="compose")

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.write_videofile(str(output_path), fps=fps, codec="libx264")

    # Clean up
    for clip in clips:
        clip.close()
    final.close()

    return str(output_path)
