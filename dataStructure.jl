mutable struct NodeList
    id::Int64
    label::Int64
    distance::Float64
    next::NodeList

    function NodeList()
        x = new()
        x.next = x
    end

    function NodeList(id, label, distance)
        x = new()
        x.id = id
        x.label = label
        x.distance = distance
        x.next = x
        return x
    end

    function NodeList(id, label, distance, next)
        x = new()
        x.id = id
        x.label = label
        x.distance = distance
        x.next = next
        return x
    end
end

function push_in_list(list::NodeList, id, label, distance)
    new_NodeList = NodeList(id, label, distance, list)
    return new_NodeList
end

function ordered_push_in_list(list::NodeList, id, label, distance)
    past_position = list
    current_position = list
    last_point = list
    while(current_position.next != current_position.next.next && current_position.distance < distance)
        past_position = current_position
        current_position = current_position.next
    end
    if(list == list.next)
        new_NodeList = NodeList(id, label, distance, list)
        return new_NodeList
    end

    if(distance >= current_position.distance)
        new_NodeList = NodeList(id, label, distance, current_position.next)
        current_position.next = new_NodeList
    else
        if(list.next == list.next.next || current_position == list)
            new_NodeList = NodeList(id, label, distance, current_position)
            return new_NodeList
        else
            new_NodeList = NodeList(id, label, distance, current_position)
            past_position.next = new_NodeList
        end
    end
    return list
end

function ordered_push_in_listO1(list::NodeList, id, label, distance, k)
    past_position = list
    current_position = list
    last_point = list
    counter = 1
    while(current_position.next != current_position.next.next && current_position.distance < distance)
        past_position = current_position
        current_position = current_position.next
        counter = counter + 1
    end
    if counter > k + 1
        return list
    end
    if(list == list.next)
        new_NodeList = NodeList(id, label, distance, list)
        return new_NodeList
    end
    if(distance >= current_position.distance)
        new_NodeList = NodeList(id, label, distance, current_position.next)
        current_position.next = new_NodeList
    else
        if(list.next == list.next.next || current_position == list)
            new_NodeList = NodeList(id, label, distance, current_position)
            return new_NodeList
        else
            new_NodeList = NodeList(id, label, distance, current_position)
            past_position.next = new_NodeList
        end
    end
    return list
end

function print_list(list::NodeList)
    last_point = list
    current_position = list
    while(current_position != current_position.next)
        println(current_position.id, " ", current_position.label, " ", current_position.distance, "-> Imprimindo a Lista")
        current_position = current_position.next
    end
end
