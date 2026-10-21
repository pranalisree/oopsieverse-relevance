"""
List every damageable object in a task scene, its type, distance to anchor
objects, and the relevance a config assigns it. Use it to write relevance YAMLs.

    python -m relevance.inspect_scene --task dishes_to_sink
    python -m relevance.inspect_scene --task dishes_to_sink --relevance s3
    python -m relevance.inspect_scene --task counter_to_microwave --anchors coffee_cup ref:microwave
"""

from __future__ import annotations

import argparse


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True)
    p.add_argument("--relevance", default=None, help="relevance config name or path")
    p.add_argument("--anchors", nargs="*", default=None,
                   help="keys to measure distance from (default: objects with rho=1 in --relevance, else the robot)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from relevance.env import make_raw_env
    from relevance.weights import RelevanceConfig, load_relevance_config, match_key, object_distance, object_type

    rel = load_relevance_config(args.relevance, task=args.task) if args.relevance else RelevanceConfig.uniform(0.0)
    env = make_raw_env(args.task, rel, track="all", seed=args.seed, verbose_relevance=bool(args.relevance))
    env.reset()
    objects = env.unique_damageable_objects()

    if args.anchors:
        anchors = [o for k in args.anchors for o in match_key(env, objects, k)]
    elif args.relevance:
        anchors = [o for o in objects if env.relevance.get(o.name, 0) >= 1.0 and not env._is_robot(o)]
    else:
        anchors = [o for o in objects if env._is_robot(o)]
    anchor_names = {a.name for a in anchors}

    refs, object_ids = {}, {id(o) for o in objects}
    for attr, val in vars(env).items():
        if id(val) in object_ids:
            refs.setdefault(val.name, []).append(f"ref:{attr}")

    rows = []
    for obj in objects:
        dist = min((object_distance(env, a, obj) for a in anchors if a is not obj), default=float("nan"))
        rows.append((dist if obj.name not in anchor_names else -1.0, obj))
    rows.sort(key=lambda r: (r[0], r[1].name))

    print(f"\n{args.task}: {len(objects)} damageable objects; anchors = {sorted(anchor_names)}")
    print(f"{'dist(m)':>8s}  {'rho':>5s}  {'name':45s} {'type':20s} {'parts':>5s}  refs")
    for dist, obj in rows:
        d = "anchor" if dist < 0 else f"{dist:.3f}"
        robot = " (robot)" if env._is_robot(obj) else ""
        print(f"{d:>8s}  {env.relevance.get(obj.name, 0):5g}  {obj.name:45s} {object_type(obj):20s} "
              f"{len(obj.damageable_links):5d}  {' '.join(refs.get(obj.name, []))}{robot}")
    env.close()


if __name__ == "__main__":
    main()
