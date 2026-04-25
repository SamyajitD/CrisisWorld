from CrisisWorld.models import EnvConfig, NoOp, DeployResource
from CrisisWorld.server import CrisisWorld as CrisisWorldEnvironment
from CrisisWorld.models import Observation, ActionUnion

def logger(count, obs: Observation, action: ActionUnion | None) -> None:
    if action:
        print(f"[Action Taken]: {action.model_dump(exclude_none=True)}")
    else:
        print("Initial Observation:")
    print(f"turn={count}, done={obs.done}, reward={obs.reward}")
    for reg in obs.regions:
        print(
            f"{reg.region_id} pop={reg.population} "
            f"inf={reg.infected} rec={reg.recovered} "
            f"dec={reg.deceased}"
        )
    print("-" * 40)
    print()

env = CrisisWorldEnvironment(config=EnvConfig(max_turns=5, num_regions=4))
obs = env.reset(seed=42, episode_id="manual-test")
logger(0, obs, None)

# Turn 1 (NoOp)
action = NoOp()
obs = env.step(action)
logger(1, obs, action)
# Turn 2 (NoOp)
action = NoOp()
obs = env.step(action)
logger(2, obs, action)
# Turn 3 (DeployResource to r0)
action = DeployResource(resource="medical", region_id="r0", amount=10)
obs = env.step(action)
logger(3, obs, action)
# Turn 4 (NoOp)
action = NoOp()
obs = env.step(action)
logger(4, obs, action)
# Turn 5 (NoOp)
action = DeployResource(resource="medical", region_id="r0", amount=100)
# action = NoOp()
obs = env.step(action)
logger(5, obs, action)