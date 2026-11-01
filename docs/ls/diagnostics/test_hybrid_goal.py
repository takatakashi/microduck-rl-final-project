from dataclasses import replace
from types import SimpleNamespace
import unittest
from hybrid_goal import SwitchingController, validate_pair, select_goal_seeds, experiment_scenario
from microduck_project.controller import Observation, Criteria


class SwitchingTests(unittest.TestCase):
    def setUp(self):
        self.controller=SwitchingController((.6,0),max_forward=.4,max_yaw_rate=1.)
        self.obs=Observation(.49,0,0,.1,.3,0.)

    def test_switch_and_hysteresis_without_state_writes(self):
        self.assertEqual(self.controller.update(self.obs,1).policy,'stand')
        self.assertEqual(self.obs.x,.49)
        self.assertEqual(self.controller.update(replace(self.obs,x=.44),2).policy,'stand')
        self.assertEqual(self.controller.update(replace(self.obs,x=.40),3).policy,'walk')

    def test_near_target_with_motion_is_not_success(self):
        self.controller.update(self.obs,1)
        result=self.controller.update(self.obs,10)
        self.assertFalse(result.success)
        self.assertEqual(result.held_seconds,0)
        self.assertEqual(result.twist,(0.,0.,0.))

    def test_original_two_second_hold_and_velocity_thresholds(self):
        stopped=replace(self.obs,speed=.04,yaw_rate=.1)
        self.controller.update(stopped,1)
        self.assertFalse(self.controller.update(stopped,2.99).success)
        self.assertTrue(self.controller.update(stopped,3).success)

    def test_movement_or_radius_exit_resets_hold(self):
        stopped=replace(self.obs,speed=0,yaw_rate=0)
        for interrupted in [self.obs,replace(stopped,x=.44),replace(stopped,yaw_rate=.2)]:
            c=SwitchingController((.6,0),max_forward=.4,max_yaw_rate=1.)
            c.update(stopped,0);c.update(interrupted,1)
            self.assertEqual(c.update(stopped,2).held_seconds,0)
            self.assertFalse(c.update(stopped,3).success)

    def test_timeout_fall_nonfinite_and_time_reversal(self):
        for obs,reason in [(self.obs,'timeout'),(replace(self.obs,tilt=1.1),'fall'),
                           (replace(self.obs,x=float('nan')),'non_finite_state')]:
            c=SwitchingController((.6,0),max_forward=.4,max_yaw_rate=1.)
            result=c.update(obs,30)
            self.assertFalse(result.success);self.assertEqual(result.reason,reason)
        self.controller.update(self.obs,2)
        with self.assertRaises(ValueError):self.controller.update(self.obs,1)

    def test_incompatible_models_rejected(self):
        from microduck_project.common import read_json,ROOT
        exp=read_json(ROOT/'configs/experiments/baseline.json')
        meta={k:'same' for k in ['upstream_commit','lock_sha256','source_sha256','configuration_sha256','io_contract']}
        meta['experiment']=exp
        validate_pair(meta,dict(meta))
        for key in ['source_sha256','configuration_sha256','io_contract']:
            with self.assertRaises(ValueError):validate_pair(meta,{**meta,key:'different'})

    def test_validation_seeds_are_fixed_disjoint_and_do_not_mutate_scenario(self):
        scenario={'goal_seeds':[300,301,302],'selection_seeds':[100,101,102],'final_seeds':list(range(200,210))}
        self.assertEqual(select_goal_seeds(scenario,True),list(range(400,410)))
        self.assertEqual(select_goal_seeds(scenario,False),[300,301,302])
        self.assertEqual(scenario['goal_seeds'],[300,301,302])
        scenario['goal_seeds'].append(400)
        with self.assertRaises(ValueError):select_goal_seeds(scenario,True)

    def test_speed_experiment_changes_only_distance_gain_and_excludes_final_seeds(self):
        from microduck_project.common import read_json,ROOT
        scenario=read_json(ROOT/'configs/evaluation.json')
        self.assertEqual(experiment_scenario(scenario),scenario)
        fast=experiment_scenario(scenario,fast_approach=True)
        self.assertEqual(fast['goal_seeds'],[300,301,302])
        self.assertEqual(fast['goal_criteria']['distance_gain'],1.2)
        self.assertEqual(scenario['goal_criteria']['distance_gain'],.8)
        fast['goal_criteria']['distance_gain']=.8
        self.assertEqual(fast,scenario)
        with self.assertRaises(ValueError):experiment_scenario(scenario,validation=True,fast_approach=True)

    def test_fast_approach_increases_command_within_trained_range(self):
        for distance in [1.,.5,.4,.3,.2,.121]:
            with self.subTest(distance=distance):
                obs=Observation(.6-distance,0,0,.1,0,0)
                normal=SwitchingController((.6,0),max_forward=.4,max_yaw_rate=1.)
                fast=SwitchingController((.6,0),max_forward=.4,max_yaw_rate=1.,criteria=Criteria(distance_gain=1.2))
                old,new=normal.update(obs,0),fast.update(obs,0)
                self.assertLessEqual(new.twist[0],.4)
                self.assertGreaterEqual(new.twist[0],old.twist[0])
                if distance<.5:self.assertGreater(new.twist[0],old.twist[0])
                self.assertEqual(new.twist[1:],old.twist[1:])

    def test_fast_approach_keeps_alignment_stop_and_two_second_hold(self):
        fast=SwitchingController((.6,0),max_forward=.4,max_yaw_rate=1.,criteria=Criteria(distance_gain=1.2))
        turn=Observation(0,0,1.,0,0,0)
        self.assertEqual(fast.update(turn,0).twist[0],0)
        moving=self.obs
        self.assertEqual(fast.update(moving,1).policy,'stand')
        stopped=replace(moving,speed=.04,yaw_rate=.1)
        self.assertEqual(fast.update(stopped,2).held_seconds,0)
        self.assertFalse(fast.update(stopped,3.99).success)
        result=fast.update(stopped,4)
        self.assertTrue(result.success)
        self.assertEqual(result.twist,(0.,0.,0.))

    def test_fast_validation_changes_only_seeds_from_the_successful_pilot(self):
        from microduck_project.common import read_json,ROOT
        scenario=read_json(ROOT/'configs/evaluation.json')
        pilot=experiment_scenario(scenario,fast_approach=True)
        final=experiment_scenario(scenario,fast_validation=True)
        self.assertEqual(final['goal_seeds'],list(range(500,510)))
        used=set(scenario['selection_seeds']+scenario['final_seeds']+pilot['goal_seeds']+list(range(400,410)))
        self.assertFalse(used.intersection(final['goal_seeds']))
        final['goal_seeds']=pilot['goal_seeds']
        self.assertEqual(final,pilot)
        for flag in ['validation','fast_approach']:
            with self.assertRaises(ValueError):experiment_scenario(scenario,fast_validation=True,**{flag:True})
        scenario['goal_seeds'].append(500)
        with self.assertRaises(ValueError):experiment_scenario(scenario,fast_validation=True)


if __name__=='__main__':unittest.main()
