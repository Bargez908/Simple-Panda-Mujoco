import mujoco_py
import numpy as np
import random
import json
import os 
import glfw


''' 
DOCUMENTATION: 
https://github.com/openai/mujoco-py/blob/4830435a169c1f3e3b5f9b58a7c3d9c39bdf4acb/mujoco_py/generated/wrappers.pxi


'''

    # self._sim.model.actuator_ctrlrange
    # self._sim.model.actuator_names

    # self._sim.model.jnt_range
    # self._sim.model.joint_names

    # self._sim.model.site_range
    # self._sim.model.site_names
  

class MjEnv(object):

    def __init__(
        self, 
        env_name,
        controller=None,   
        specs=None, 
        init_joint_config=None, 
        max_episode_length=None 
        ):
        '''
        --- arguments ---

        @env_name: (string) name of an environment   
        @init_joint_config: if None it starts from model defined default position | joints (list) starts from a specific joints config | "random" (string) starts from random position
        @folder_path: (string) path to the folder where the xml file is contained
        @episode_terminator: instance of class EpisodeTerminator with method done(state, time) that returns a True when end the current episode

        --- variables ---

        state: (list) RL environment states (specification in variable states_specification)
        action: (list) agent action (specification in variable actions_specification) 
        done: (bool) episode termination
        states_shape: (tuple) states shape
        states_type: (tuple) states type
        actions_shape: (tuple) actions shape
        actions_type: (tuple) actions type
        states_specification: (dict) {'shape': (tuple) states_shape, 'type': (str) states_type}
        actions_specification: (dict)('shape': (tuple) actions_shape, 'type': (str) actions_type}


        Example:

            env = MjEnv(
                env_name="ur5",  
                max_episode_length=5000,
                init_joint_config=[0, 0, 0, 0, 0, 0])

            actions = [0,0,0,0,0,0]
            state, done = env.execute(actions)
            env.render()

        '''

        self.env_name = env_name
        self.controller = controller
        self.specs = specs
        
        env_data_folder = "assets"
 
        ##### SPECIFICATIONS #### 
        param_path = os.path.join(env_data_folder, self.env_name, "specs.json")  
  
        with open(param_path, 'r') as fi:
            param_spec = json.loads(fi.read()) 
            self._states_specs = param_spec["states"] 
            self._actions_specs = param_spec["actions"] 
            self._env_params = param_spec["environment"] 

        num_states = np.sum([int(sdata["dim"]) for sname, sdata in self._states_specs.items()], dtype=int)  
        self.state_shape = (num_states,) 
        num_actions = len(list(self._actions_specs.keys()))
        self.action_shape = (num_actions,)  

        self.states_specification = dict(shape=tuple(self.state_shape), type="float")
        self.actions_specification = dict(shape=tuple(self.action_shape), type="float")
 

        ##### MODEL #### 
        self.xml_path = os.path.join(env_data_folder, self.env_name, "arena_reach.xml")  
        self._mjmodel = mujoco_py.load_model_from_path(self.xml_path)

        ##### SIMULATOR ####

        # sim & ctrl frequencies
        self.simulation_frequency = 1./self._mjmodel.opt.timestep
        
        self.control_frequency = self._env_params["control_frequency"]
        nsubsteps = round(self.simulation_frequency/self._env_params["control_frequency"])
        assert nsubsteps > 0, f"expected 'control_frequency'({self.control_frequency})<='simulation_frequency'({self.simulation_frequency}) " 

        self._sim = mujoco_py.MjSim(self._mjmodel, nsubsteps=nsubsteps)

        # viewer 
        self._viewer = None
    
        # starting position
        self._init_joint_config = init_joint_config
        if self._init_joint_config is not None:
            self._set_robot_joints(self._init_joint_config)

        # --- support variables --
        self._render_ct = 0
        self._episode_max_time = max_episode_length
        self._site_forced = {}
        self._body_forced = {}

        # --- public variables ---
        self.fixed_frame_name = self._env_params["fixed_frame"] 
        self.fixed_frame = self.env_fixed_frame(self.fixed_frame_name)
        self.episode_time = 0
        self.episode_index = 0
        self.state = self.get_state()  # state
        self.action = None  # action 
        self.done = False
 
    def _refresh(self):
        for name,pos in self._site_forced.items():
            self.set_site_pos(name,pos) 
        for name,pos in self._body_forced.items():
            self.set_body_pos(name,pos) 
 
    def execute(self, action):
        ''' Takes a sorted list of actions (float), that are the torques to the motors'''

        self.action = action 

        for i in range(len(action)):
            self._sim.data.ctrl[i] = self.action[i]
        
        self._sim.step()

        self._refresh()
        self.episode_time += 1
        self.state = self.get_state() 
        if self._episode_max_time is not None:
            if self.episode_time >= self._episode_max_time:
                self.done = True
                return self.state, self.done 
          
        return self.state, self.done
 

    def reset(self, hard_reset=True, initi_pos=None, initi_displace=None, reload_model=False):
        ''' reset the episode simulation.
        The robot default initial position is the one specified in the constructor, with a displacement `initi_displace` if give.
        Otherwise it can start from a new position `initi_pos`'''
        if reload_model:
            self._mjmodel = mujoco_py.load_model_from_path(self.xml_path)
        if hard_reset:  
            nsubsteps = round(self.simulation_frequency/self._env_params["control_frequency"])
            self._sim = mujoco_py.MjSim(self._mjmodel, nsubsteps=nsubsteps)
            if self._viewer is not None:
                glfw.destroy_window(self._viewer.window) 
                self._viewer = mujoco_py.MjViewer(self._sim)

            
        self.episode_index += 1
        self.episode_time = 0
        self.done = False
        self._sim.reset()
        
        if initi_pos is not None:
            self._set_robot_joints(initi_pos)
        elif self._init_joint_config is not None:
            initi_pos = self._init_joint_config
            if initi_displace is not None:
                initi_pos = list(np.array(initi_pos) + np.array(initi_displace))

            self._set_robot_joints(initi_pos)

    """
    def get_contact_forces(self):
        print('number of contacts', self._sim.data.ncon)
        for i in range(self._sim.data.ncon):
            # Note that the contact array has more than `ncon` entries,
            # so be careful to only read the valid entries.
            contact = self._sim.data.contact[i]
            print('contact', i)
            print('dist', contact.dist)
            print('geom1', contact.geom1, self._sim.model.geom_id2name(contact.geom1))
            print('geom2', contact.geom2, self._sim.model.geom_id2name(contact.geom2))
            # There's more stuff in the data structure
            # See the mujoco documentation for more info!
            geom2_body = self._sim.model.geom_bodyid[self._sim.data.contact[i].geom2]
            print('Contact force on geom2 body', self._sim.data.cfrc_ext[geom2_body])
            print('norm', np.sqrt(np.sum(np.square(self._sim.data.cfrc_ext[geom2_body]))))
            # Use internal functions to read out mj_contactForce
            c_array = np.zeros(6, dtype=np.float64)
            mujoco_py.functions.mj_contactForce(self._sim.model, self._sim.data, i, c_array)
            print('c_array', c_array)
            self.render()
        print("ritorno")
        return c_array[0]
    """
            

    def render(self, episode_step=0, do_first=False):
        ''' render and visualize 1 time step of the simulation. Must be called in a loop.
            episode_step is the number of episodes to skip'''
 
        # we create the viewer in the first render call
        if self._viewer is None:
            self._viewer = mujoco_py.MjViewer(self._sim)

        if self.done:
            self._render_ct += 1
            if self._render_ct > int(episode_step):
                self._render_ct = 0

        if self._render_ct == int(episode_step) or (do_first and self.episode_index == 0):
            self._viewer.render()
            return True
        else:
            return False
  
    def _set_robot_joints(self, joints): # BUG controllare dove usato... 
        joint_ranges = self._sim.model.jnt_range
        joint_names = self._sim.model.joint_names 
        for i, name in enumerate(joint_names):   
            joint_index = self._sim.model.get_joint_qpos_addr(name)
            r = joint_ranges[joint_names.index(name)]
            if joints == 'random':  
                jval = random.uniform(r[0], r[1])
            else:
                jval = joints[i]
                if jval<r[0]:
                    jval = r[0]
                if jval>r[1]:
                    jval = r[1] 
            self._sim.data.qpos[joint_index] = jval
            if i>=len(self._init_joint_config)-1:    #needed to avoid over jointing
               break
        self._sim.forward()
        # self._sim.step()

    def set_site_pos(self, name, pos):
        self._site_forced[name] = pos
        site_id = self._sim.model.site_name2id(name)
        self._sim.data.site_xpos[site_id] = pos

    '''def set_body_pos(self, name, pos):       # BUG: non funziona
        self._body_forced[name] = pos
        body_id = self._sim.model.body_name2id(name)
        self._sim.data.body_xpos[body_id] = pos
        print("set body pos", name, pos)
    ''' 
    def set_joint_pos(self, name, pos):
        self._sim.data.set_joint_qpos(name, pos)
        
    def get_obj_pos(self, objname, objtype="site", objvar="xpos"):  
        attr_get_value = getattr(self._sim.data,f"get_{objtype}_{objvar}") 
        pos = attr_get_value(objname)  
        return pos

    def get_joints_pos(self, ids=None, names=None):
        if names is not None and ids is None:
            joints_pos = self._sim.data.get_joint_qpos(names)
        elif ids is None:
            joints_pos = self._sim.data.qpos
        else:
            joints_pos = self._sim.data.qpos[ids]
        #print("cube pos2", self._sim.data.get_joint_qpos("cube:joint"))
        return joints_pos

    def get_joints_vel(self, ids=None):   
        if ids is None:
            joints_vel = self._sim.data.qvel
        else:
            joints_vel = self._sim.data.qvel[ids]
        return joints_vel
    
    def get_joints_acc(self, ids=None):   
        if ids is None:
            joints_acc = self._sim.data.qacc
        else:
            joints_acc = self._sim.data.qacc[ids]
        return joints_acc
    """
    def get_state(self):
        ''' return the states from the given list (states_list) '''
        state = [] 
        sim_states = self._sim.get_state()

        ###########################
        print("njoints", self._mjmodel.njnt, "end of njoints")
        #n_joints = self._mjmodel.njnt
        #print("n joints", n_joints, "end of n joints")
        # Loop over the joints and print their names
        #print ("joint id", self._mjmodel.jnt_qposadr[1], "end of joint id")
        #print("joint name", self._mjmodel.joint_id2name(1), "end of joint name")
        #print("body names", self._mjmodel.body_names, "end of body names")
        #print("joint qpos", self._sim.data.qpos, "end of joint qpos")
        #print("body xpos",self._sim.data.body_xpos, "end of body xpos") #ordinati come i nomi
        #print("body quat",self._sim.data.body_xquat,"end of body quat") #ordinati come i nomi
        ###########################

        for sname, sdata in self._states_specs.items():

            sid = sdata["id"]   
            stype = sdata["type"]     
            svar = sdata["var"]     

            attr_get_value = getattr(self._sim.data,f"get_{stype}_{svar}") 
            sval = attr_get_value(sid)  
            if stype in ["body","site"]: 
                sval = list(np.array(sval)-np.array(self.fixed_frame[:3]))  # convert wrt fixed frame
                # TODO rotazione!!!!  

            sval = list(np.array(sval).flatten())   

            state.append(sval)
        state = [item for sublist in state for item in sublist]  
        return state
   """
    
    def get_state(self):
        state = {}

         # Get the positions and velocities of all joints
        joint_pos = self._sim.data.qpos
        joint_vel = self._sim.data.qvel
        state["joint_pos"] = joint_pos
        state["joint_vel"] = joint_vel
        site_pos = self._sim.data.site_xpos
        state['site_pos'] = site_pos
        site_vel = self._sim.data.site_xvelp
        state['site_vel'] = site_vel
        #print("stati",state,"end stati")
        #print("joint_pos", joint_pos, "end joint pos")
        #print("joint_vel", joint_vel, "end joint vel")
        #print("site pos", site_pos, "end site pos")
        #print("site vel", site_vel, "end site vel")

        return state

    def env_fixed_frame(self, name=None):
        ''' returns the fixed reference frame as a list [x,y,z,qw,qx,qy,qz]'''
        frame = []
        if name is None:
            frame = [0, 0, 0, 1, 0, 0, 0]
        else:
            pos = self._sim.data.get_body_xpos(name)
            quat = self._sim.data.get_body_xquat(name)
            for p in pos:
                frame.append(p)
            for q in quat:
                frame.append(q)
        self.fixed_frame = frame
        return frame
