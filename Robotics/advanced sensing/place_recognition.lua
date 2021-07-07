-- Generate a sample from a Gaussian distribution
function gaussian (mean, variance)
    return  math.sqrt(-2 * variance * math.log(math.random() + 0.00001)) *
            math.cos(2 * math.pi * math.random()) + mean
end


-- Move robot to a location (only for use in random setup, not from your code!)
function setRobotPose(handle, x, y, theta)
    allModelObjects = sim.getObjectsInTree(handle) -- get all objects in the model
    sim.setThreadAutomaticSwitch(false)
    for i=1,#allModelObjects,1 do
        sim.resetDynamicObject(allModelObjects[i]) -- reset all objects in the model
    end
    pos = sim.getObjectPosition(handle, -1)
    sim.setObjectPosition(handle, -1, {x, y, pos[3]})
    sim.setObjectOrientation(handle, -1, {0, 0, theta})
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



function get_walls()
    -- Disable error reporting
    local savedState=sim.getInt32Parameter(sim.intparam_error_report_mode)
    sim.setInt32Parameter(sim.intparam_error_report_mode,0)
    local N = 1
    while true do
        local handle = sim.getObjectHandle("Wall"..tostring(N))
        if handle <= 0 then
            break
        end

        -- Read position and shape of wall
        -- Assume here that it is thin and oriented either along the x axis or y axis

        -- We can now get the propertries of these walls, e.g....
        local pos = sim.getObjectPosition(handle, -1)
        local res,minx = sim.getObjectFloatParameter(handle,15)
        local res,maxx = sim.getObjectFloatParameter(handle,18)
        local res,miny = sim.getObjectFloatParameter(handle,16)
        local res,maxy = sim.getObjectFloatParameter(handle,19)
    
        --print("Position of Wall " .. tostring(N) .. ": " .. tostring(pos[1]) .. "," .. tostring(pos[2]) .. "," .. tostring(pos[3]))
        --print("minmax", minx, maxx, miny, maxy)
 
        local Ax, Ay, Bx, By
        if (maxx - minx > maxy - miny) then
            print("Wall " ..tostring(N).. " along x axis")
            Ax = pos[1] + minx
            Ay = pos[2]
            Bx = pos[1] + maxx
            By = pos[2]
        else
            print("Wall " ..tostring(N).. " along y axis")
            Ax = pos[1]
            Ay = pos[2] + miny
            Bx = pos[1]
            By = pos[2] + maxy
        end
        print (Ax, Ay, Bx, By)

        walls[N] = {Ax, Ay, Bx, By}
        N = N + 1
    end
    -- enable error reporting
    sim.setInt32Parameter(sim.intparam_error_report_mode,savedState)

    return N - 1
end


function get_goals()
    -- Disable error reporting
    local savedState=sim.getInt32Parameter(sim.intparam_error_report_mode)
    sim.setInt32Parameter(sim.intparam_error_report_mode,0)
    local N = 1
    goalHandles = {}
    while true do
        local handle = sim.getObjectHandle("Goal"..tostring(N))
        if handle <= 0 then
            break
        end
        
        goalHandles[N] = handle

        -- Read position of goal
        local pos = sim.getObjectPosition(handle, -1)
    
        print("Position of Goal " .. tostring(N) .. ": " .. tostring(pos[1]) .. "," .. tostring(pos[2]) .. "," .. tostring(pos[3]))

        goals[N] = {pos[1], pos[2]}
        N = N + 1
    end
    -- enable error reporting
    sim.setInt32Parameter(sim.intparam_error_report_mode,savedState)

    return N - 1
end



-- Robot should call this function when it thinks it has reached goal N
-- Second argument is the robot's handle
function reachedGoal(N, handle)

    green = {0, 1, 0}
    yellow = {1, 1, 0}
    blue = {0, 0, 1}
    grey = {0.5, 0.5, 0.5}



    
    local pos = sim.getObjectPosition(handle, -1)
    local xerr = pos[1] - goals[N][1]
    local yerr = pos[2] - goals[N][2]
    local err = math.sqrt(xerr^2 + yerr^2)
    local localpts = 0
    local colour = grey
    if (err < 0.05) then
        localpts = 3
        colour = green
    elseif (err < 0.1) then
        localpts = 2
        colour = yellow
    elseif (err < 0.2) then
        localpts = 1
        colour = blue
    end
 
    -- Colour the goal
    --local goalHandle = sim.getObjectHandle("Goal" .. tostring(N))
    sim.setShapeColor(goalHandles[N], nil, sim.colorcomponent_ambient_diffuse, colour)
 
    -- if we're not at final goal (which is where we started)
    if (localpts > 0 and goalsReached[N] == false) then
        goalsReached[N] = true
        totalPoints = totalPoints + localpts
        print ("Reached Goal" ..tostring(N).. " with error " ..tostring(err).. ": Points: " ..tostring(localpts))
    end

    -- at final goal: have we reached all goals?
    if (N == startGoal and localpts > 0) then
        local allGoalsReached = true
        for i=1,N_GOALS do
            if (goalsReached[i] == false) then
                allGoalsReached = false
            end
        end
        -- Yes... all goals achieved so calculate time
        if (allGoalsReached == true) then
            tt = sim.getSimulationTime() 
            timeTaken = tt - startTime
            timePoints = 0
            if (timeTaken < 60) then
                timePoints = 5
            elseif (timeTaken < 90) then
                timePoints = 4
            elseif (timeTaken < 120) then
                timePoints = 3
            elseif (timeTaken < 180) then
                timePoints = 2
            elseif (timeTaken < 240) then
                timePoints = 1
            end
            totalPoints = totalPoints + timePoints
            print ("FINISH at time" ..tostring(timeTaken).. " with total points " ..tostring(totalPoints))

            sim.pauseSimulation()
        end
    end

end

function get_Signiture_for_place(goal_num)
    setRobotPose(robotBase, goals[goal_num][1], goals[goal_num][2], 0.0)  --start from absolute 0 angle
    --sim.setObjectOrientation(robotBase, -1, {0, 0, 0})
    idx = 0
    for angle=-180, 180, 10 do
        idx = idx + 1
        --setRobotPose(robotBase, goals[goal_num][1], goals[goal_num][2], math.rad(angle))
        sim.setJointTargetPosition(turretMotor, angle)
        --now take depth measurements
        result,cleanDistance=sim.readProximitySensor(turretSensor)
        if (result>0) then
            noisyDistance = cleanDistance + gaussian(0.0, sensorVariance)
        else
            noisyDistance = 0.0
        end
        learned_places_depthdata[goal_num][idx] = noisyDistance
    end
    
    --convert histogram to invariant signiture
    invariant_signiture = {}
    for i=1, #learned_places_depthdata do
        invariant_signiture[i] = {}
        for k=1, (1/bin_width) do
            invariant_signiture[i][k] = 0
        end
        idx = 0
        for k=0, 1, bin_width do
            idx = idx + 1
            for j=1, #learned_places_depthdata[i] do
                if (learned_places_depthdata[i][j]>=k) and (learned_places_depthdata[i][j]<(k+bin_width)) then
                    invariant_signiture[i][idx] = invariant_signiture[i][idx] + 1
                end
            end
        end
    end
    
    --reset robot to original random position
    sim.setJointTargetPosition(turretMotor, startOrientation)
    setRobotPose(robotBase, startx, starty, startOrientation)
    print('histogram data: \n', learned_places_depthdata)
    print('invariant data: \n', invariant_signiture, '\n has dimension of: ', (#invariant_signiture)*(#invariant_signiture[1]))
    return invariant_signiture
end



-- This function is executed exactly once when the scene is initialised
function sysCall_init()

    startTime = sim.getSimulationTime()
    print("Start Time", startTime)
          
    robotBase=sim.getObjectAssociatedWithScript(sim.handle_self) -- robot handle
    leftMotor=sim.getObjectHandle("leftMotor") -- Handle of the left motor
    rightMotor=sim.getObjectHandle("rightMotor") -- Handle of the right motor
    turretMotor=sim.getObjectHandle("turretMotor") -- Handle of the turret motor
    turretSensor=sim.getObjectHandle("turretSensor")
     
    -- Please use noisyDistance= cleanDistance + gaussian(0.0, sensorVariance) for all sonar sensor measurements
    sensorStandardDeviation = 0.01
    sensorVariance = sensorStandardDeviation^2
    noisyDistance = 0

    -- Create bumpy floor for robot to drive on
    createRandomBumpyFloor()
 
    -- Data structure for walls  (your program can use this)
    walls = {}
    -- Fill it by parsing the scene in the GUI
    N_WALLS = get_walls()
    -- walls now is an array of arrays with the {Ax, Ay, Bx, By} wall coordinates



    -- Data structure for goals (your program can use this)
    goals = {}
    -- Fill it by parsing the scene in the GUI
    N_GOALS = get_goals()
    -- goals now is an array of arrays with the {Gx, Gy} goal coordinates
    
    learned_places_depthdata = {}

    for g=1,N_GOALS do
        print ("Goal" ..tostring(g).. " Gx " ..tostring(goals[g][1]).. " Gy " ..tostring(goals[g][2]))
        learned_places_depthdata[g] = {}
    end
    
    
 

    -- Randomise robot start position to one of the goals with random orientation
    startGoal = math.random(N_GOALS)
    startx = goals[startGoal][1]
    starty = goals[startGoal][2]
    startOrientation = math.random() * 2 * math.pi
    setRobotPose(robotBase, startx, starty, startOrientation)
 
 
    -- These variables are for keeping score, and they will be changed by reachedGoal() --- don't change them directly!
    totalPoints = 0
    goalsReached = {}
    for i=1,N_GOALS do
        goalsReached[i] = false
    end
  
    --invariant bin width
    bin_width = 0.01
       
    -- Your code here!
    
    -- EXAMPLE: student thinks they have reached a goal
    -- reachedGoal(1, robotBase)

    
    
end

function sysCall_sensing()
    
end

function get_squared_diff(lst_1, lst_2)
    sum = 0
    for i=1, #lst_1 do
        sum = sum + (lst_1[i] - lst_2[i])^2
    end
    return sum
end

--[[function placeRecognition(current_pos)
    shift = 0.0
    while (shift <= 360) do
        shift = shift + 0.5   --increment the shift in degree
        new_angle = (current_pos[3] - math.rad(shift))
        if new_angle<(-math.rad(180)) then
            new_angle = new_angle + math.rad(360)
        elseif new_angle>math.rad(180) then
            new_angle = new_angle - math.rad(360)
        end
        setRobotPose(robotBase, current_pos[1], current_pos[2], new_angle)
        test_depthdata = {}
        idx = 0
        for angle=0, 360, 10 do
            idx = idx + 1
            setRobotPose(robotBase, current_pos[1], current_pos[2], math.rad(angle))
            --now take depth measurements
            result,cleanDistance=sim.readProximitySensor(turretSensor)
            if (result>0) then
                noisyDistance = cleanDistance + gaussian(0.0, sensorVariance)
            else
                noisyDistance = 0.0
            end
            test_depthdata[idx] = noisyDistance
        end
        --print(test_depthdata)
        --now compare with the 5 learned places
        lowest_diff = 1e109
        best_fit = 0
        for i=1, #learned_places_depthdata do
            --compute sum of squared diff between depth histograms
            sum_sqdiff = get_squared_diff(test_depthdata, learned_places_depthdata[i])
            if (sum_sqdiff < lowest_diff) then
                lowest_diff = sum_sqdiff
                best_fit = i
            end
        end
        best_place = {best_fit, lowest_diff, shift}
        print(best_place)
        --now check if the a similarity threshold has been met with the shift
        if (best_place[2] <= 0.1) then
            print('localization completed: ', best_place)
            return best_place
        end
    end
    --print('the best localization result is: ', best_place)
end]]--


function placeRecognition(data_signiture)
    test_depthdata = {}
    idx = 0
    for angle=-180, 180, 10 do
        idx = idx + 1
        sim.setJointTargetPosition(turretMotor, angle)
        --now take depth measurements
        result,cleanDistance=sim.readProximitySensor(turretSensor)
        if (result>0) then
            noisyDistance = cleanDistance + gaussian(0.0, sensorVariance)
        else
            noisyDistance = 0.0
        end
        test_depthdata[idx] = noisyDistance
    end
    print('test histogram is: \n', test_depthdata)
    --convert to invariant test signiture
    invariant_signiture = {}
    for k=1, (1/bin_width) do
        invariant_signiture[k] = 0
    end
    idx = 0
    for k=0, 1, bin_width do
        idx = idx + 1
        for j=1, #test_depthdata do
            if (test_depthdata[j]>=k) and (test_depthdata[j]<(k+bin_width)) then
                invariant_signiture[idx] = invariant_signiture[idx] + 1
            end
        end
    end
    print('test invariant signiture is: \n', invariant_signiture)
    --now compare with the 5 learned places
    lowest_diff = 1e109
    best_fit = 0
    for i=1, #data_signiture do
        --compute sum of squared diff between invariant signitures
        sum_sqdiff = get_squared_diff(invariant_signiture, data_signiture[i])
        if (sum_sqdiff < lowest_diff) then
            lowest_diff = sum_sqdiff
            best_fit = i
        end
    end
    best_place = {best_fit, lowest_diff}
    print('the found best guess is: ', best_place)
    --now check if the a similarity threshold has been met with the shift
    if (best_place[2] <= 0.1) then
        print('localization completed: ', best_place)
        return best_place
    end
    --print('the best localization result is: ', best_place)
end

function sysCall_actuation() 
    tt = sim.getSimulationTime()
    -- print("actuation hello", tt)
    
    if (tt-startTime) < 0.1 then
        for g=1,N_GOALS do
            signiture = get_Signiture_for_place(g)
        end
        --print('table has number of entries: ', (#learned_places_depthdata)*(#learned_places_depthdata[1]), '\n', learned_places_depthdata)
    end
    
    placeRecognition(signiture)

    print('current pos: ', sim.getObjectPosition(robotBase, -1))
    
    --[[current_pos = sim.getObjectPosition(robotBase, -1)
    test_depthdata = {}
    idx = 0
    for angle=0, 360, 10 do
        idx = idx + 1
        setRobotPose(robotBase, current_pos[1], current_pos[2], math.rad(angle))
        --now take depth measurements
        result,cleanDistance=sim.readProximitySensor(turretSensor)
        if (result>0) then
            noisyDistance = cleanDistance + gaussian(0.0, sensorVariance)
        else
            noisyDistance = 0.0
        end
        test_depthdata[idx] = noisyDistance
    end
    lowest_diff = 1e109
    best_fit = 0
    for i=1, #learned_places_depthdata do
        --compute sum of squared diff between depth histograms
        sum_sqdiff = get_squared_diff(test_depthdata, learned_places_depthdata[i])
        if (sum_sqdiff < lowest_diff) then
            lowest_diff = sum_sqdiff
            best_fit = i
        end
    end
    best_place = {best_fit, lowest_diff}
    
    print(best_place)]]--
    
    --localization = placeRecognition(sim.getObjectPosition(robotBase, -1))
    --print('current localized goal number is: ', localization)

    result,cleanDistance=sim.readProximitySensor(turretSensor)
    if (result>0) then
        noisyDistance= cleanDistance + gaussian(0.0, sensorVariance)
        print ("Depth sensor reading ", noisyDistance)
    end


    -- Your code here!

end

function sysCall_cleanup()
    for g=1,N_GOALS do
        sim.setShapeColor(goalHandles[g], nil, sim.colorcomponent_ambient_diffuse, {1, 0, 0})
    end
end 
