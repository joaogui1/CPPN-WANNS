import hashlib
import os
import time
import random
import subprocess as sub

RUN_DIR = "basic_data"

def read_voxlyze_results(filename="softbotsOutput.xml"):
    i = 0
    max_attempts = 60
    file_size = 0
    this_file = ""

    while (i < max_attempts) and (file_size == 0):
        try:
            file_size = os.stat(filename).st_size
            this_file = open(filename)
            this_file.close()
        except FileNotFoundError:
            file_size = 0
        i += 1
        time.sleep(1)

    if file_size == 0:
        print(f"file {filename} wasn't present")
        return 0
    
    fitness = 0
    try:
        with open(filename) as fitfile:
            for line in fitfile:
                if "<FinalDist" in line:
                    fitness = float(line[line.find("<FinalDist") + len("<FinalDist>"):line.find("</" + "FinalDist")])
    except:
        ls_check = sub.check_output(["ls", RUN_DIR + "/fitnessFiles/"], encoding='utf-8').split()
        print(ls_check, f"\n{filename}\n{filename in ls_check}")
    return fitness


def write_voxelyze_file(sim, env, individual, run_directory, run_name):

    voxelyze_file = open(run_directory + "/voxelyzeFiles/" + run_name + "--id_%s.vxa" % individual.id, "w")

    voxelyze_file.write(
        "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n\
        <VXA Version=\"1.0\">\n\
        <Simulator>\n")

    # Sim
    for name, tag in sim.new_param_tag_dict.items():
        voxelyze_file.write(tag + str(getattr(sim, name)) + "</" + tag[1:] + "\n")

    voxelyze_file.write(
        "<Integration>\n\
        <Integrator>0</Integrator>\n\
        <DtFrac>" + str(sim.dt_frac) + "</DtFrac>\n\
        </Integration>\n\
        <Damping>\n\
        <BondDampingZ>1</BondDampingZ>\n\
        <ColDampingZ>0.8</ColDampingZ>\n\
        <SlowDampingZ>0.01</SlowDampingZ>\n\
        </Damping>\n\
        <Collisions>\n\
        <SelfColEnabled>" + str(int(sim.self_collisions_enabled)) + "</SelfColEnabled>\n\
        <ColSystem>3</ColSystem>\n\
        <CollisionHorizon>2</CollisionHorizon>\n\
        </Collisions>\n\
        <Features>\n\
        <FluidDampEnabled>0</FluidDampEnabled>\n\
        <PoissonKickBackEnabled>0</PoissonKickBackEnabled>\n\
        <EnforceLatticeEnabled>0</EnforceLatticeEnabled>\n\
        </Features>\n\
        <SurfMesh>\n\
        <CMesh>\n\
        <DrawSmooth>1</DrawSmooth>\n\
        <Vertices/>\n\
        <Facets/>\n\
        <Lines/>\n\
        </CMesh>\n\
        </SurfMesh>\n\
        <StopCondition>\n\
        <StopConditionType>" + str(int(sim.stop_condition)) + "</StopConditionType>\n\
        <StopConditionValue>" + str(sim.simulation_time) + "</StopConditionValue>\n\
        <AfterlifeTime>" + str(sim.afterlife_time) + "</AfterlifeTime>\n\
        <MidLifeFreezeTime>" + str(sim.mid_life_freeze_time) + "</MidLifeFreezeTime>\n\
        <InitCmTime>" + str(sim.fitness_eval_init_time) + "</InitCmTime>\n\
        </StopCondition>\n\
        <EquilibriumMode>\n\
        <EquilibriumModeEnabled>" + str(sim.equilibrium_mode) + "</EquilibriumModeEnabled>\n\
        </EquilibriumMode>\n\
        <GA>\n\
        <WriteFitnessFile>1</WriteFitnessFile>\n\
        <FitnessFileName>" + run_directory + "/fitnessFiles/softbotsOutput--id_%s.xml" % individual.id +
        "</FitnessFileName>\n\
        <QhullTmpFile>" + run_directory + "/tempFiles/qhullInput--id_%s.txt" % individual.id + "</QhullTmpFile>\n\
        <CurvaturesTmpFile>" + run_directory + "/tempFiles/curvatures--id_%s.txt" % individual.id +
        "</CurvaturesTmpFile>\n\
        </GA>\n\
        <MinTempFact>" + str(sim.min_temp_fact) + "</MinTempFact>\n\
        <MaxTempFactChange>" + str(sim.max_temp_fact_change) + "</MaxTempFactChange>\n\
        <MaxStiffnessChange>" + str(sim.max_stiffness_change) + "</MaxStiffnessChange>\n\
        <MinElasticMod>" + str(sim.min_elastic_mod) + "</MinElasticMod>\n\
        <MaxElasticMod>" + str(sim.max_elastic_mod) + "</MaxElasticMod>\n\
        <ErrorThreshold>" + str(0) + "</ErrorThreshold>\n\
        <ThresholdTime>" + str(0) + "</ThresholdTime>\n\
        <MaxKP>" + str(0) + "</MaxKP>\n\
        <MaxKI>" + str(0) + "</MaxKI>\n\
        <MaxANTIWINDUP>" + str(0) + "</MaxANTIWINDUP>\n")

    voxelyze_file.write("</Simulator>\n")

    # Env
    voxelyze_file.write(
        "<Environment>\n")
    for name, tag in env.new_param_tag_dict.items():
        voxelyze_file.write(tag + str(getattr(env, name)) + "</" + tag[1:] + "\n")

    voxelyze_file.write(
        "<Fixed_Regions>\n\
        <NumFixed>0</NumFixed>\n\
        </Fixed_Regions>\n\
        <Forced_Regions>\n\
        <NumForced>0</NumForced>\n\
        </Forced_Regions>\n\
        <Gravity>\n\
        <GravEnabled>" + str(env.gravity_enabled) + "</GravEnabled>\n\
        <GravAcc>-9.81</GravAcc>\n\
        <FloorEnabled>" + str(env.floor_enabled) + "</FloorEnabled>\n\
        <FloorSlope>" + str(env.floor_slope) + "</FloorSlope>\n\
        </Gravity>\n\
        <Thermal>\n\
        <TempEnabled>" + str(env.temp_enabled) + "</TempEnabled>\n\
        <TempAmp>" + str(env.temp_amp) + "</TempAmp>\n\
        <TempBase>25</TempBase>\n\
        <VaryTempEnabled>1</VaryTempEnabled>\n\
        <TempPeriod>" + str(1.0 / env.frequency) + "</TempPeriod>\n\
        </Thermal>\n\
        <TimeBetweenTraces>" + str(env.time_between_traces) + "</TimeBetweenTraces>\n\
        <StickyFloor>" + str(env.sticky_floor) + "</StickyFloor>\n\
        </Environment>\n")

    voxelyze_file.write(
        "<VXC Version=\"0.93\">\n\
        <Lattice>\n\
        <Lattice_Dim>" + str(env.lattice_dimension) + "</Lattice_Dim>\n\
        <X_Dim_Adj>1</X_Dim_Adj>\n\
        <Y_Dim_Adj>1</Y_Dim_Adj>\n\
        <Z_Dim_Adj>1</Z_Dim_Adj>\n\
        <X_Line_Offset>0</X_Line_Offset>\n\
        <Y_Line_Offset>0</Y_Line_Offset>\n\
        <X_Layer_Offset>0</X_Layer_Offset>\n\
        <Y_Layer_Offset>0</Y_Layer_Offset>\n\
        </Lattice>\n\
        <Voxel>\n\
        <Vox_Name>BOX</Vox_Name>\n\
        <X_Squeeze>1</X_Squeeze>\n\
        <Y_Squeeze>1</Y_Squeeze>\n\
        <Z_Squeeze>1</Z_Squeeze>\n\
        </Voxel>\n\
        <Palette>\n\
        <Material ID=\"1\">\n\
            <MatType>0</MatType>\n\
            <Name>Passive_Soft</Name>\n\
            <Display>\n\
            <Red>0</Red>\n\
            <Green>1</Green>\n\
            <Blue>1</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(env.fat_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>1e+006</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>0</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        <Material ID=\"2\">\n\
            <MatType>0</MatType>\n\
            <Name>Passive_Hard</Name>\n\
            <Display>\n\
            <Red>0</Red>\n\
            <Green>0</Green>\n\
            <Blue>1</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(env.bone_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>1e+006</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>0</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
            <Material ID=\"3\">\n\
            <MatType>0</MatType>\n\
            <Name>Active_+</Name>\n\
            <Display>\n\
            <Red>1</Red>\n\
            <Green>0</Green>\n\
            <Blue>0</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(env.muscle_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>1e+006</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>" + str(0.01*(1+random.uniform(0, env.actuation_variance))) + "</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        <Material ID=\"4\">\n\
            <MatType>0</MatType>\n\
            <Name>Active_-</Name>\n\
            <Display>\n\
            <Red>0</Red>\n\
            <Green>1</Green>\n\
            <Blue>0</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(env.muscle_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>1e+006</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>" + str(-0.01*(1+random.uniform(0, env.actuation_variance))) + "</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        <Material ID=\"5\">\n\
            <MatType>0</MatType>\n\
            <Name>Obstacle</Name>\n\
            <Display>\n\
            <Red>1</Red>\n\
            <Green>0.784</Green>\n\
            <Blue>0</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>5e+007</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>1e+006</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>0</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        <Material ID=\"6\">\n\
            <MatType>0</MatType>\n\
            <Name>Head_Active_+</Name>\n\
            <Display>\n\
            <Red>1</Red>\n\
            <Green>1</Green>\n\
            <Blue>0</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(env.fat_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>1e+006</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>" + str(0.01 * (1 + random.uniform(0, env.actuation_variance))) + "</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        <Material ID=\"7\">\n\
            <MatType>0</MatType>\n\
            <Name>Food</Name>\n\
            <Display>\n\
            <Red>1</Red>\n\
            <Green>1</Green>\n\
            <Blue>0</Blue>\n\
            <Alpha>1</Alpha>\n\
            </Display>\n\
            <Mechanical>\n\
            <MatModel>0</MatModel>\n\
            <Elastic_Mod>" + str(env.muscle_stiffness) + "</Elastic_Mod>\n\
            <Plastic_Mod>0</Plastic_Mod>\n\
            <Yield_Stress>0</Yield_Stress>\n\
            <FailModel>0</FailModel>\n\
            <Fail_Stress>0</Fail_Stress>\n\
            <Fail_Strain>0</Fail_Strain>\n\
            <Density>1e+006</Density>\n\
            <Poissons_Ratio>0.35</Poissons_Ratio>\n\
            <CTE>0</CTE>\n\
            <uStatic>1</uStatic>\n\
            <uDynamic>0.5</uDynamic>\n\
            </Mechanical>\n\
        </Material>\n\
        </Palette>\n\
        <Structure Compression=\"ASCII_READABLE\">\n\
        <X_Voxels>" + str(individual.orig_size[0]) + "</X_Voxels>\n\
        <Y_Voxels>" + str(individual.orig_size[1]) + "</Y_Voxels>\n\
        <Z_Voxels>" + str(individual.orig_size[2]) + "</Z_Voxels>\n")

    
    voxelyze_file.write("<Data>\n")
    
    for z in range(individual.orig_size[2]):
        voxelyze_file.write("<Layer><![CDATA[")
        voxelyze_file.write("".join(individual.phenotype[z]))
        # for y in range(individual.orig_size[1]):
            # for x in range(individual.orig_size[0]):
                # voxelyze_file.write(individual.phenotype[z][y*individual.orig_size[0] + x])
        voxelyze_file.write("]]></Layer>\n")
    
    voxelyze_file.write("</Data>\n")

    # append custom parameters
    # string_for_md5 = ""

    voxelyze_file.write(
        "</Structure>\n\
        </VXC>\n\
        </VXA>")
    voxelyze_file.close()

    # m = hashlib.md5()
    # m.update(string_for_md5.encode('utf-8'))

    # return m.hexdigest()
