-- Generate a sample from a Gaussian distribution
function gaussian (mean, variance)
    return  math.sqrt(-2 * variance * math.log(math.random())) *
            math.cos(2 * math.pi * math.random()) + mean
end


-- Return robot to a location
function resetBase(handle, matrix)
    allModelObjects = sim.getObjectsInTree(handle) -- get all objects in the model
    sim.setThreadAutomaticSwitch(false)
    for i=1,#allModelObjects,1 do
        sim.resetDynamicObject(allModelObjects[i]) -- reset all objects in the model
    end
    sim.setObjectMatrix(handle,-1,matrix)
    sim.setThreadAutomaticSwitch(true)
end

function createRandomBumpyFloor()
    print ("Generating new random bumpy floor.")
    sim.setThreadAutomaticSwitch(false)

    -- Remove existing bumpy floor if there already is one
    if (heightField ~= nil) then
        sim.setObjectPosition(heightField, heightField, {0.05, 0, 0})
        return
    end
    --  Create random bumpy floor for robot to drive on
    floorSize = 5
    --heightFieldResolution = 0.3
    --heightFieldNoise = 0.00000005
    heightFieldResolution = 0.1
    heightFieldNoise = 0.0000008
    cellsPerSide = floorSize / heightFieldResolution
    cellHeights = {}
    for i=1,cellsPerSide*cellsPerSide,1 do
        table.insert(cellHeights, gaussian(0, heightFieldNoise))
    end
    heightField=sim.createHeightfieldShape(0, 0, cellsPerSide, cellsPerSide, floorSize, cellHeights)
    -- Make the floor invisible
    sim.setObjectInt32Parameter(heightField,10,0)
    sim.setThreadAutomaticSwitch(true)
end


-- This function is executed exactly once when the scene is initialised
function sysCall_init()

    tt = sim.getSimulationTime()
    print("Init hello", tt)
          
    robotBase=sim.getObjectAssociatedWithScript(sim.handle_self) -- robot handle
    leftMotor=sim.getObjectHandle("leftMotor") -- Handle of the left motor
    rightMotor=sim.getObjectHandle("rightMotor") -- Handle of the right motor
    turretMotor=sim.getObjectHandle("turretMotor") -- Handle of the turret motor
    turretSensor=sim.getObjectHandle("turretSensor")
    trajectoryGraph=sim.getObjectHandle("trajectoryGraph")
    sensingGraph=sim.getObjectHandle("sensingGraph")
     
    -- We only update graphs every few steps because the simulation slows down otherwise
    UPDATE_GRAPHS_EVERY = 1
    graphSteps = 0


    
 

    -- Create bumpy floor for robot to drive on
    createRandomBumpyFloor()

    -- Save robot start position so we can return it there later
    robotStartMatrix=sim.getObjectMatrix(robotBase,-1)
   
    -- Usual rotation rate for wheels (radians per second)
    speedBase = 5
    
    -- Which step are we in? 
    -- 0 is a dummy value which is immediately completed
    stepCounter = 0
    stepCompletedFlag = false
    stepList = {}
        

    --[[stepList[1] = {"forward_until_bump"}
    stepList[2] = {"stop"}
    stepList[3] = {"turn_random", math.rad(90)}
    stepList[4] = {"stop"}
    stepList[5] = {"repeat"}]]--
    
    stepList[1] = {"start_wall_following", 1.00}
    stepList[2] = {"stop"}
    stepList[3] = {"repeat"}
 
 

    -- Target positions for joints
    motorAngleTargetL = 0.0
    motorAngleTargetR = 0.0


    -- To calibrate
    motorAnglePerMetre = 24.5
    motorAnglePerRadian = 3.1
 
 
    sensorStandardDeviation = 0.03
    sensorVariance = sensorStandardDeviation^2

    noisyDistance = 0
    --set random bump threshold
    min_dist = 0.25
    
    --set wall following proportional control
    kp = 4
    initial_turret_angle = math.rad(70)
end

function sysCall_sensing()
    
end


function isCurrentTargetAchieved(posL, posR)

    -- Start with returnVal = true and negate it if any parts of target are not reached
    returnVal = true
    if (speedBaseL > 0 and posL < motorAngleTargetL) then
        returnVal = false
    end
    if (speedBaseL < 0 and posL > motorAngleTargetL) then
        returnVal = false
    end
    if (speedBaseR > 0 and posR < motorAngleTargetR) then
        returnVal = false
    end
    if (speedBaseR < 0 and posR > motorAngleTargetR) then
        returnVal = false
    end

    return returnVal
end


function sysCall_actuation() 
    tt = sim.getSimulationTime()
    -- print("actuation hello", tt)

    -- Get and plot current angles of motor joints
    posL = sim.getJointPosition(leftMotor)
    posR = sim.getJointPosition(rightMotor)
    
    
    graphSteps = graphSteps + 1



    if graphSteps % UPDATE_GRAPHS_EVERY == 0 then
        sim.handleGraph(sim.handle_all, tt+sim.getSimulationTimeStep())
    end
 

    result,cleanDistance=sim.readProximitySensor(turretSensor)

    if (result>0) then
        noisyDistance= cleanDistance + gaussian(0.0, sensorVariance)

        -- Add the data to the graph:
        sim.setGraphUserData(sensingGraph,"ND",noisyDistance)
    end



    -- Start new step?
    
    if (stepCompletedFlag == true or stepCounter == 0) then
        stepCounter = stepCounter + 1
        stepCompletedFlag = false


        newStepType = stepList[stepCounter][1]

        if (newStepType == "repeat") then
            -- Loop back to the first step
            stepCounter = 1
            newStepType = stepList[stepCounter][1]
        end
 


        print("New step:", stepCounter, newStepType)
 
        --[[if (newStepType == "forward_until_bump") then
            print ("New forward step")
            speedBaseL = speedBase
            speedBaseR = speedBase
        elseif (newStepType == "turn_random") then
            print ("Starting new random turn")
            --print ("Need to fill in code here")
            newStepAmount = stepList[stepCounter][2]
            speedBaseL = -speedBase
            speedBaseR = speedBase
            random_turn = gaussian(0.0, 0.2)
            motorAngleTargetL = posL - newStepAmount * motorAnglePerRadian * (1 + math.abs(random_turn))
            motorAngleTargetR = posR + newStepAmount * motorAnglePerRadian * (1 + math.abs(random_turn))
        elseif (newStepType == "stop") then
            speedBaseL = 0
            speedBaseR = 0
        end]]--
        
        if (newStepType == "start_wall_following") then
            print ("start following")
            --set turret sensor to always 70 degrees to the left wrt. the forward robot direction
            robot_pos = sim.getObjectPosition(robotBase, -1)
            sim.setJointTargetPosition(turretMotor, (robot_pos[3]+initial_turret_angle))
            
            --newStepAmount = stepList[stepCounter][2]
            --speedBaseL = speedBase
            --speedBaseR = speedBase
            --motorAngleTargetL = posL + newStepAmount * motorAnglePerMetre - 0.5*kp*(noisyDistance - min_dist)
            --motorAngleTargetR = posR + newStepAmount * motorAnglePerMetre + 0.5*kp*(noisyDistance - min_dist)
        elseif (newStepType == "stop") then
            speedBaseL = 0
            speedBaseR = 0
        end
    end

    -- Set the motor velocities for the current step
    --sim.setJointTargetVelocity(leftMotor,speedBaseL)
    --sim.setJointTargetVelocity(rightMotor,speedBaseR)        



    -- Handle current ongoing step
    stepType = stepList[stepCounter][1]
    
    if (stepType == "start_wall_following") then  
        print ("Handling following")
        --print ("Need to fill in code here and set stepCompletedFlag = true when finished")
        result,cleanDistance=sim.readProximitySensor(turretSensor)
        if (result>0) then
            noisyDistance= cleanDistance + gaussian(0.0, sensorVariance)
            -- Add the data to the graph:
            sim.setGraphUserData(sensingGraph,"ND",noisyDistance)
        end
        --set new wheel angular speeds
        speedBaseL = speedBase - 0.5*kp*(noisyDistance - min_dist)
        speedBaseR = speedBase + 0.5*kp*(noisyDistance - min_dist)
        sim.setJointTargetVelocity(leftMotor,speedBaseL)
        sim.setJointTargetVelocity(rightMotor,speedBaseR)
        --never stops
        stepCompletedFlag = false
        --print ("noisyDistance =", noisyDistance)
    elseif (stepType == "stop") then
        -- Check to see if the robot is stationary to within a small threshold
        linearVelocity,angularVelocity=sim.getVelocity(robotBase)
        vLin = math.sqrt(linearVelocity[1]^2 + linearVelocity[2]^2 + linearVelocity[3]^2)
        vAng = math.sqrt(angularVelocity[1]^2 + angularVelocity[2]^2 + angularVelocity[3]^2)
        --print ("stop", linearVelocity, vLin, vAng)
    
        if (vLin < 0.001 and vAng < 0.01) then
            stepCompletedFlag = true
        end
    end

    --[[if (stepType == "forward_until_bump") then  
        print ("Handling forward_until_bump")
        --print ("Need to fill in code here and set stepCompletedFlag = true when finished")
        if (noisyDistance<min_dist) then
            stepCompletedFlag = true
        end
        --print ("noisyDistance =", noisyDistance)
    elseif (stepType == "turn_random") then
        print ("Handling turn_random")
        --print ("Need to fill in code here")
        if isCurrentTargetAchieved(posL, posR) then
            stepCompletedFlag = true
        end
    elseif (stepType == "stop") then
        -- Check to see if the robot is stationary to within a small threshold
        linearVelocity,angularVelocity=sim.getVelocity(robotBase)
        vLin = math.sqrt(linearVelocity[1]^2 + linearVelocity[2]^2 + linearVelocity[3]^2)
        vAng = math.sqrt(angularVelocity[1]^2 + angularVelocity[2]^2 + angularVelocity[3]^2)
        --print ("stop", linearVelocity, vLin, vAng)
    
        if (vLin < 0.001 and vAng < 0.01) then
            stepCompletedFlag = true
        end
    end]]--


    
end

function sysCall_cleanup()
    --simUI.destroy(ui)
end 