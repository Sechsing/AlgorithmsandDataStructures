### DO NOT CHANGE THIS FUNCTION
def load_dictionary(filename):
    infile = open(filename)
    word, frequency = "", 0
    aList = []
    for line in infile:
        line.strip()
        if line[0:4] == "word":
            line = line.replace("word: ", "")
            line = line.strip()
            word = line
        elif line[0:4] == "freq":
            line = line.replace("frequency: ", "")
            frequency = int(line)
        elif line[0:4] == "defi":
            index = len(aList)
            line = line.replace("definition: ", "")
            definition = line.replace("\n", "")
            aList.append([word, definition, frequency])

    return aList

from typing import List, Tuple, Optional, Union

class Node:
    def __init__(self, data: Tuple[Optional[str], Optional[str], Optional[int]] = (None, None, None), child_num: int = 27) -> None:
        """
        Function description:
        Sets up a Node object in a Trie.

        :Input:
        data: A tuple containing word details such as text, meaning, and frequency.
        child_num: An integer that determines the array size for child nodes.
        :Output, return or postcondition:  Produces a Node with attributes such as word, definition, frequency,
        node_frequency, and child.
        :Time complexity: O(1)
        :Aux space complexity: O(child_num) where child_num is the number of elements in self.child list
        """
        self.word = data[0]
        self.definition = data[1]
        self.frequency = data[2]
        self.node_frequency = 0
        self.child = [None] * child_num

class Trie:
    def __init__(self, Dictionary: List[List[Union[str, int]]]) -> None:
        """
        Function description:
        Constructs a Trie using the provided dictionary data.

        :Input:
        Dictionary: A nested list where each inner list contains the word's details.
        :Output, return or postcondition: Establishes a Trie with a root node and fills it using the dictionary data.
        :Time complexity:
        O(T), where T is the total number of characters in Dictionary, as each character results in a Node creation.
        :Aux space complexity:
        O(T), where T is the total number of characters in Dictionary, as each character results in a Node creation.
        """
        self.root = Node()
        for words in Dictionary:
            self.insert_word(words[0], words)

    def insert_word(self, word: str, data: List[Union[str, int]]) -> None:
        """
        Function description:
        Adds a word and its details to the Trie.

        Approach description (if main function):
        The method begins at the root node, updates node_frequency, and checks/stores data. Then, it iteratively
        processes each word character.

        :Input:
        word: The word to be added.
        data: A List with word details: word text, meaning, and frequency.
        :Output, return or postcondition: Places the word and its details into the Trie.
        :Time complexity: O(M*min(X, Y)), where M*min(X, Y) is the time complexity of insert_word_aux function.
        :Aux space complexity: O(M), where M is the aux space complexity of insert_word_aux function.
        """
        node = self.root
        # Increment the node frequency of the root node
        node.node_frequency += 1
        # Check and store data in the current node
        self.update_best_word(node, data)
        # Call auxiliary method
        self.insert_word_aux(node, word, 0, data)

    def insert_word_aux(self, node: Node, word: str, word_index: int, data: Optional[List[Union[str, int]]] = None) -> None:
        """
        Function description:
        Auxiliary function for inserting a word and its details into the Trie.

        Approach description (if main function):
        The function recursively inserts a word and its details. For each character, it determines the child node index
        using the character's alphabet position. If the child node already exists, it moves to that node. Otherwise, a
        new node is created. The node_frequency is updated at each node, and the word details are checked. Once the word
        end is reached, the word and its details are stored in the first child node.

        :Input:
        current: The current node in the Trie.
        word: The word to be added.
        word_index: The current index of the word.
        data: A list containing word information: word, definition, frequency.
        :Output, return or postcondition: Recursively inserts the word and its details.
        :Time complexity: O(M*min(X, Y)), where M is the length of the word and the function is called M times
        recursively to insert each character in the word and in the process compare method is performed that has
        O(min(X, Y)) time complexity.
        :Aux space complexity: O(M), where M is the length of the word and a new Node is created for each word character.
        """
        # Store the data in the first node when there is no more characters in the word
        if word_index == len(word):
            node.child[0] = Node(data)
            node = node.child[0]
        else:
            # The character at the current position
            char = word[word_index]
            # Calculate the index based on the character's position in the alphabet
            index = ord(char)-97+1
            if node.child[index] is not None:
                # Move to the existing child node
                node = node.child[index]
            else:
                # Create a new child node if it does not exist
                node.child[index] = Node()
                node = node.child[index]
            # Increment the node_frequency
            node.node_frequency += 1
            # Call the update_best_word method to update word information
            self.update_best_word(node, data, word_index)
            # Recursively insert the word
            self.insert_word_aux(node, word, word_index+1, data)

    def update_best_word(self, node: Node, data: List[Union[str, int]], word_index: int = 0) -> None:
        """
        Function description:
        Compares a new word's frequency with an existing word at a node and updates accordingly.

        Approach description (if main function):
        The function compares the frequency of a newly inserted word with the frequency of the word already stored at a
        node and updates the node with the word in data if data and current.frequency is not None and the word in data
        has a higher frequency. If both have the same frequency the function will compare the words alphabetically to
        determine which is smaller to store. It does this by increasing the index each time until the characters are
        different.

        :Input:
        current: The current node in the Trie.
        data: A list containing word details such as word, definition, frequency.
        :Output, return or postcondition: Updates the node with the word and its details based on frequency and
        alphabetical order.
        :Time complexity: O(min(X, Y)), where X is the number of characters in the current node's word and Y is the
        number of character in the data's word. The function compares the order of character in current node's word and
        data's word min(X, Y) times to determine if it will replace current node with data.
        :Aux space complexity: O(1)
        """
        if data is not None and node.frequency is not None:
            if data[2] > node.frequency:
                # Replace current node's data with the data if data's frequency is higher
                node.word, node.definition, node.frequency = data
            elif data[2] == node.frequency:
                # Compare the order of characters while the strings have characters and the characters are the same
                while word_index < len(data[0]) and word_index < len(node.word) and data[0][word_index] == node.word[
                    word_index]:
                    word_index += 1
                if word_index < len(data[0]) and word_index < len(node.word) and data[0][word_index] < node.word[word_index]:
                    # Replace current with data if data is alphabetically smaller
                    node.word, node.definition, node.frequency = data
        else:
            # Set the current node's data to the data if data or current.frequency is None
            node.word, node.definition, node.frequency = data

    def prefix_search(self, prefix: str) -> List[Union[str, int]]:
        """
        Function description:
        Searches for a word in the Trie by its prefix.

        Approach description (if main function):
        The prefix_search method begins at the root node and then recursively searches using a auxiliary function.

        :Input:
        prefix: The prefix for the search.
        :Output, return or postcondition: Returns a list with the word, its definition, and the node frequency for the
        matched prefix.
        :Time complexity: O(M), where M is the time complexity of prefix_search_aux method.
        :Aux space complexity: O(1)
        """
        node = self.root
        # Call auxiliary method
        return self.prefix_search_aux(node, prefix, 0)

    def prefix_search_aux(self, node: Node, prefix: str, prefix_index: int) -> List[Union[str, int]]:
        """
        Function description:
        Auxiliary function to recursively search a prefix in the Trie.

        Approach description (if main function):
        The prefix_search_aux method is responsible for performing a prefix search recursively within the Trie. For each
        prefix character, it moves to the corresponding child node if it exists. If not, it indicates a non-match. The
        process continues until the prefix's end, where it returns the node's details. If the node doesn't exist, it
        returns [None, None, 0] to indicate that no matching word was found. At the end of the prefix, it returns the
        information of the node: word, definition, and node frequency, which represents the words in the Trie that share
        the prefix and has the highest frequency.

        :Input:
        current: The current node in the Trie.
        prefix: The prefix for the search.
        prefix_index: The current index of the prefix.
        :Output, return or postcondition: Returns a list containing word, definition, and node_frequency for the
        matched prefix.
        :Time complexity: O(M), where M is the length of the prefix and the function is performed M times recursively
        to search each character of the prefix.
        :Aux space complexity: O(1)
        """
        # Check if the prefix has been completely processed
        if prefix_index == len(prefix):
            return [node.word, node.definition, node.node_frequency]
        else:
            char = prefix[prefix_index]
            index = ord(char)-97+1
            # Progress to the child node for the current character if it is present
            if node.child[index]:
                node = node.child[index]
            # Return default values if there is no corresponding child node for the character,
            else:
                return [None, None, 0]
            # Proceed with the recursive search for the next character
            return self.prefix_search_aux(node, prefix, prefix_index+1)

import math

class Vertex:
    def __init__(self, type: str, id: int) -> None:
        """
        Function description:
        Initializes a Vertex object with a given type and ID.

        :Input:
        type: The type of the vertex, typically representing its role or category in the network, for example: "Person",
        "Driver", "Car", "Source", "Sink".
        id: A unique identifier for the vertex.
        :Output, return or postcondition:
        Creates a Vertex object with the provided type and ID, and initializes an empty list for edges.
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        self.type = type
        self.id = id
        self.edges = []

    def __str__(self):
        """
        Function description:
        Returns a string representation of the Vertex object.

        :Input: None
        :Output, return or postcondition:
        Returns a string that combines the type and ID of the vertex.
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        return self.type + " " + str(self.id)

class Edge:
    def __init__(self, start: Vertex, end: Vertex, capacity: int) -> None:
        """
        Function description:
        Initializes an Edge object that represents a directed edge between two vertices in the flow network with a given
        capacity.

        :Input:
        start: The starting vertex of the edge.
        end: The ending vertex of the edge.
        capacity: The maximum allowable flow through the edge.
        :Output, return or postcondition:
        Creates an Edge object with the specified start and end vertices, and capacity. Also initializes the flow to 0
        and reverse edge reference to None.
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        self.start = start
        self.end = end
        self.capacity = capacity
        self.flow = 0
        self.reverse = None

    def __str__(self):
        """
        Function description:
        Returns a string representation of the Edge object.

        :Input: None
        :Output, return or postcondition:
        Returns a string detailing the start and end vertices, and the flow and capacity of the edge.
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        return self.start.type + " " + str(self.start.id) + " -> " + self.end.type + " " + str(self.end.id) + " | Flow / Capacity: " + str(self.flow) + "/" + str(self.capacity)

class FlowNetwork:
    def __init__(self, preferences: List[List[int]], licenses: List[int]) -> None:
        """
        Function description:
        Initializes the FlowNetwork object based on given preferences and licenses.

        :Input:
        preferences: A list of lists where each inner list indicates the preference of a person for car allocation.
        licenses: A list indicating which persons are licensed drivers.

        :Output, return or postcondition:
        Initializes the vertices of the flow network and constructs the flow network using the provided preferences and licenses.

        :Time complexity: O(N^2), where N^2 is the time complexity of the initialize method.
        :Aux space complexity: O(N), where N is the number of persons in preferences list.
        """
        self.vertices = []
        # Initialize the flow network using preferences and licenses
        self.initialize(preferences, licenses)

    def initialize(self, preferences: List[List[int]], licenses: List[int]) -> None:
        """
        Function description:
        Sets up the initial flow network structure using the given preferences and licenses.

        :Input:
        preferences: A list of lists where each inner list indicates the preference of a person for car allocation.
        licenses: A list indicating which persons are licensed drivers.

        :Output, return or postcondition:
        Constructs the flow network with source, sink, person vertices, driver vertices, and connections based on preferences and licenses.

        :Time complexity: O(N^2), where N is the number of persons in preferences. This is because the function calls
        multiple other functions to add and connect vertices, the process iterates over nested loops of preferences
        which has a N number of persons hence the O(N^2) complexity.
        :Aux space complexity: O(N), where N is the number of persons in preferences.
        """
        # Create source and sink vertices
        source = Vertex("Source", 0)
        sink = Vertex("Sink", 0)
        self.vertices.extend([source, sink])
        # Add person vertices and connect them to the source
        person_list = self.add_person(preferences)
        # Add driver vertices and connect them to the sink
        driver_list = self.add_driver(preferences)
        # Connect persons to drivers based on preferences and licenses
        self.connect_person_to_driver(person_list, driver_list, preferences, licenses)
        # Add car vertices
        car_list = self.add_car(preferences)
        # Connect persons to cars based on preferences
        self.connect_person_to_car(person_list, car_list, preferences)
        # Add a bridge vertex and connect car vertices to it
        self.add_bridge(car_list, preferences)

    def add_person(self, preferences: List[List[int]]) -> List[Vertex]:
        """
        Function description:
        Creates and connects person vertices to the source vertex.

        :Input:
        preferences: A list of lists where each inner list indicates the preference of a person for car allocation.
        :Output, return or postcondition:
        Returns a list of all person vertices created.
        :Time complexity: O(N), where N is the number of persons in preferences.
        :Aux space complexity: O(N), where N is the number of persons in preferences.
        """
        # Create and return a list of person vertices
        person_list = [Vertex("Person", i) for i in range(len(preferences))]
        for person in person_list:
            # Connect each person vertex to the source vertex
            self.add_edge(self.vertices[0], person, 1)
        self.vertices.extend(person_list)
        return person_list

    def add_driver(self, preferences: List[List[int]]) -> List[Vertex]:
        """
        Function description:
        Creates driver vertices and connects them to the sink vertex.

        :Input:
        preferences: A list of lists where each inner list indicates the preference of a person for car allocation.
        :Output, return or postcondition:
        Returns a list of all driver vertices created.
        :Time complexity: O(N), where n is the number of persons in preferences list.
        :Aux space complexity: O(N), where n is the number of persons in preferences list.
        """
        # Determine the number of driver vertices needed
        num_of_driver = math.ceil(len(preferences)/5)
        # Create and return a list of driver vertices
        driver_list = [Vertex("Driver", i) for i in range(num_of_driver)]
        for driver in driver_list:
            # Connect each driver vertex to the sink vertex
            self.add_edge(driver, self.vertices[1], 2)
        self.vertices.extend(driver_list)
        return driver_list

    def connect_person_to_driver(self, person_list: List[Vertex], driver_list: List[Vertex], preferences: List[List[int]], licenses: List[int]) -> None:
        """
        Function description:
        Connects each person vertex to driver vertices based on their preferences and licenses.

        :Input:
        person_list: List of all person vertices.
        driver_list: List of all driver vertices.
        preferences: A list of lists where each inner list indicates the preference of a person for car allocation.
        licenses: A list indicating which persons are licensed drivers.
        :Output, return or postcondition:
        Modifies the flow network by adding edges between persons and drivers based on preferences and licenses.
        :Time complexity: O(N^2), where N is the number of persons in preferences list and each person's preferences is
        the number of persons divide by 5.
        :Aux space complexity: O(1).
        """
        # Loop through each person's preferences and connect them to the driver vertices
        for i, preference in enumerate(preferences):
            if i in licenses:
                for p in preference:
                    if p < len(driver_list):
                        # Connect the person vertex to the driver vertex based on preference
                        self.add_edge(person_list[i], driver_list[p], 1)

    def add_car(self, preferences: List[List[int]]) -> List[Vertex]:
        """
        Function description:
        Creates car vertices based on the number of drivers needed.

        :Input:
        preferences: A list of lists where each inner list indicates the preference of a person for car allocation.
        :Output, return or postcondition:
        Returns a list of all car vertices created.
        :Time complexity: O(N), where N is the number of persons in preferences list as one car is added for every 5
        persons.
        :Aux space complexity: O(N).
        """
        # Determine the number of car vertices needed
        num_of_driver = math.ceil(len(preferences)/5)
        # Create and return a list of car vertices
        car_list = [Vertex("Car", i) for i in range(num_of_driver)]
        self.vertices.extend(car_list)
        return car_list

    def connect_person_to_car(self, person_list: List[Vertex], car_list: List[Vertex], preferences: List[List[int]]) -> None:
        """
        Function description:
        Connects each person vertex to car vertices based on their preferences.

        :Input:
        person_list: List of all person vertices.
        car_list: List of all car vertices.
        preferences: A list of lists where each inner list indicates the preference of a person for car allocation.
        :Output, return or postcondition:
        Modifies the flow network by adding edges between persons and cars based on preferences.
        :Time complexity: O(N^2), where N is the number of persons in preferences list and each person's preferences is
        the number of persons divide by 5.
        :Aux space complexity: O(1).
        """
        # Loop through each person's preferences and connect them to the car vertices
        for i, preference in enumerate(preferences):
            for p in preference:
                if p < len(car_list):
                    # Connect the person vertex to the car vertex based on preference
                    self.add_edge(person_list[i], car_list[p], 1)

    def add_bridge(self, car_list: List[Vertex], preferences: List[List[int]]) -> None:
        """
        Function description:
        Creates a bridge vertex and connects all car vertices to this bridge. Also connects the bridge to the sink
        vertex.

        :Input:
        car_list: List of all car vertices.
        preferences: A list of lists where each inner list indicates the preference of a person for car allocation.
        :Output, return or postcondition:
        Modifies the flow network by adding edges between cars and a bridge vertex, and between the bridge vertex and
        the sink.
        :Time complexity: O(N), where N is the number of persons in preferences list as there is a connection for each
        car.
        :Aux space complexity: O(1).
        """
        # Create a bridge vertex
        bridge = Vertex("Bridge", 0)
        self.vertices.append(bridge)
        for car in car_list:
            # Connect each car vertex to the bridge vertex
            self.add_edge(car, bridge, 3)
        # Connect the bridge vertex to the sink vertex with the required capacity
        edge_capacity = len(preferences) - 2 * math.ceil(len(preferences)/5)
        self.add_edge(bridge, self.vertices[1], edge_capacity)

    def add_edge(self, start: Vertex, end: Vertex, capacity: int) -> None:
        """
        Function description:
        Creates an edge between two vertices with a given capacity and also establishes a reverse edge with zero
        capacity.

        :Input:
        start: The starting vertex for the edge.
        end: The ending vertex for the edge.
        capacity: The capacity for the edge.
        :Output, return or postcondition:
        Modifies the flow network by adding the new edge and its reverse edge.
        :Time complexity: O(1).
        :Aux space complexity: O(1).
        """
        # Create an edge between start and end vertices with the given capacity and its reverse edge
        edge = Edge(start, end, capacity)
        reverse_edge = Edge(end, start, 0)
        edge.reverse = reverse_edge
        reverse_edge.reverse = edge
        start.edges.append(edge)
        end.edges.append(reverse_edge)

    def find_vertex(self, type: str, id: int) -> Optional[Vertex]:
        """
        Function description:
        Searches for a vertex in the flow network based on its type and ID.

        :Input:
        type: The type of the vertex to be found for example: "Person", "Driver", "Car".
        id: The ID of the vertex to be found.
        :Output, return or postcondition:
        Returns the vertex if found, otherwise returns None.
        :Time complexity: O(N), where V is the number of vertices in the flow network and V evaluates to N because the
        number of vertices in the flow network is determined by the number of persons in preferences list.
        :Aux space complexity: O(1).
        """
        # Find and return a vertex based on its type and id
        for vertex in self.vertices:
            if vertex.type == type and vertex.id == id:
                return vertex

    def get_augmenting_path(self, start: Vertex, end: Vertex, path: List[Tuple[Edge, int]]) -> Optional[List[Tuple[Edge, int]]]:
        """
        Function description:
        Finds an augmenting path if it exists in the flow network between the start and end vertices.

        Approach description (if main function):
        The function employs a depth-first search (DFS) approach to identify augmenting paths in the residual graph.
        Starting from the source vertex, it traverses the flow network, examining each edge to see if there is residual
        capacity available. If there is, the method recursively explores the next vertex, appending the current edge to
        the path. The search stops when the sink vertex is reached, indicating an augmenting path has been found. If no
        such path exists, the method returns None.

        :Input:
        start: The starting vertex for the search.
        end: The ending vertex for the search.
        path: The current path being considered in the search.
        :Output, return or postcondition:
        Returns an augmenting path if one is found, otherwise returns None.
        :Time complexity: O(N) where V is the number of vertices in the flow network when it visits all the vertices of
        the flow network in the worst case scenario. O(V) evaluates to O(N) where N is the number of persons because the
        number of vertices created in the flow network is determined by the number of persons in preferences.
        :Aux space complexity: O(N) where E is the number of edges in the flow network when it calls itself recursively
        E times and stores the current path in the path list. In the worst case, the length of this list
        will be equal to the number of edges in the flow network. O(E) evaluates to O(N) where N is the number of
        persons because the number of edges created in the flow network is determined by the number of persons in
        preferences.
        """
        # Return the paths if the start vertex is the same as the end vertex
        if start == end:
            return path
        # Iterate through edges from the current start vertex
        for edge in start.edges:
            # Calculate the residual capacity of the edge
            residual_capacity = edge.capacity - edge.flow
            # If the residual capacity is positive (flow can still be added) and the edge has not been visited in the current path
            if residual_capacity > 0 and not (edge, residual_capacity) in path:
                # Recursively search for a path from the end of the current edge to the sink
                result = self.get_augmenting_path(edge.end, end, path + [(edge, residual_capacity)])
                # Return the result if a path is found
                if result != None:
                    return result

    def get_max_flow(self) -> int:
        """
        Function description:
        Computes the maximum flow from the source to the sink in the flow network using the Ford-Fulkerson algorithm.

        Approach description (if main function):
        The function uses the Ford-Fulkerson algorithm to compute the maximum flow in the flow network. It continuously
        searches for augmenting paths from the source to the sink using the get_augmenting_path method. Once an
        augmenting path is found, the function updates the flow values on the path edges based on the bottleneck
        capacity (minimum residual capacity of the path). This process is iteratively repeated until no more augmenting
        paths can be found in the residual graph. The maximum flow is then the sum of flows of all outgoing edges from
        the source.

        :Output, return or postcondition:
        Returns the value of the maximum flow from the source to the sink in the flow network.
        :Time complexity: O(N^2) where V and E are the number of vertices and edges in the flow network and the flow
        network utilizes the Ford-Fulkerson algorithm's which has a worst-case time complexity of O(VE^2) when using DFS
        to find augmenting paths. This evaluates to O(N^2) as the number of vertices and edges created in the flow
        network is determined by the number of persons in preferences list.
        :Aux space complexity: O(N) where E is the number of edges in the flow network and the function stored E number
        of paths. This evaluates to O(N) as the number of edges created in the flow network is determined by the number
        of persons in preferences list.
        """
        # Find the first augmenting path from source to sink
        path = self.get_augmenting_path(self.vertices[0], self.vertices[1], [])
        # While there exists an augmenting path in the network
        while path != None:
            # Calculate the bottleneck flow for this path (minimum residual capacity across all edges in the path)
            flow = min(edge[1] for edge in path)
            # Update the flow for each edge in the path and its corresponding reverse edge
            for edge, residue in path:
                edge.flow += flow
                edge.reverse.flow -= flow
            # Search for another augmenting path
            path = self.get_augmenting_path(self.vertices[0], self.vertices[1], [])
        # Sum the flow values of all edges coming out of the source to get the maximum flow
        return sum(edge.flow for edge in self.vertices[0].edges)

    def connections(self) -> List[List[int]]:
        """
        Function description:
        Extracts the connections or assignments of persons to cars/drivers based on the flow in the network.

        Approach description (if main function):
        The function is designed to extract the allocations of persons to cars or drivers based on the established flow
        in the network. It iteratively goes through all driver and car vertices, checking each person vertex's edges to
        see if there's a flow towards the current driver or car. If there is a flow towards a driver or car vertex, it
        indicates a connection, and the person's ID is added to the current connection list.

        :Output, return or postcondition:
        Returns a list of lists where each inner list represents a car's or driver's allocation of people.
        :Time complexity: O(N^2) where N is the number of persons in preferences list as the function iterates over
        persons and then their respective edges.
        :Aux space complexity: O(N) where N is the number of persons in preferences list and the function stores N
        number of edges in the connections list.
        """
        # Initialize an empty list to store connections of persons to cars
        connections = []
        # Calculate the number of driver vertices
        num_of_driver = math.ceil(len(self.vertices)/5)
        # Iterate over all driver vertices
        for i in range(num_of_driver):
            # Initialize an empty list to store the current connection
            connection_list = []
            # Retrieve the current driver and car vertex
            driver_vertex = self.find_vertex("Driver", i)
            car_vertex = self.find_vertex("Car", i)
            # Iterate over all vertices in the network
            for v in self.vertices:
                # If the vertex represents a person
                if v.type == "Person":
                    # Check each edge of the person vertex
                    for edge in v.edges:
                        # Assign person to driver if the edge has positive flow and leads to the current driver vertex
                        if edge.flow > 0 and edge.end == driver_vertex:
                            connection_list.append(v.id)
                        # Assign person to car if the edge has positive flow and leads to the current car vertex,
                        elif edge.flow > 0 and edge.end == car_vertex:
                            connection_list.append(v.id)
            # Add connection list to connections if it is not empty
            if connection_list:
                connections.append(connection_list)
        # Return the complete list of connections
        return connections

def allocate(preferences: List[List[int]], licenses: List[int]) -> Optional[List[List[int]]]:
    """
    Function description:
    Allocates persons to cars/drivers based on their preferences and available licenses.

    Approach description (if main function):
    The function begins by checking basic constraints like the availability of enough drivers or ensuring that each car
    has a minimum of two persons. If these constraints are not met, allocation is not possible and the function returns
    None. If the conditions are met, a flow network is set up using the FlowNetwork class, leveraging the given
    preferences and licenses. The maximum flow of this network is then computed using the get_max_flow method. If the
    maximum flow value is less than the total number of persons, it indicates that not all persons can be assigned to
    cars/drivers, and the function returns None. If the allocation is successful, the connections method is used to
    extract and return the allocations of persons to cars/drivers.

    :Input:
    preferences: A list of lists where each inner list represents the preference order of cars for a person.
    licenses: A list of integers representing the persons who have a driver's license.
    :Output, return or postcondition:
    Returns a list of nested lists detailing the assignment of individuals to cars. If such an assignment is not
    feasible, it returns None.
    :Time complexity: O(N^3), where N is the number of persons in preferences list. The time complexity is
    primarily influenced by the calculateMaxFlow function. Given that the complexity of calculateMaxFlow is O(VE^2),
    where V and E are the number of vertices and edges in the flow netork, it translates to O(N*N^2) in this context.
    This is because the number of vertices, V scales with the number of persons in preferences list. The quadratic term,
     N^2, arises from the worst-case scenario where every individual wants every
    available destination and has a license. Thus, the function has a O(N^3) time complexity.
    :Aux space complexity: O(N), where N is the aux space complexity of connections method.
    """
    # Return None when there are not enough drivers or each car does not meet the minimum requirement of 2 person
    if len(preferences) < 2 or len(licenses) < math.ceil(len(preferences)/5):
        return None
    network = FlowNetwork(preferences, licenses)
    max_flow=network.get_max_flow()
    # Return none when someone cannot be connected to a car that has 2 drivers
    if max_flow < len(preferences):
        return None
    return network.connections()

if __name__ == '__main__':
    Dictionary = load_dictionary("Dictionary.txt")
    myTrie = Trie(Dictionary)
    print(myTrie.prefix_search(""))
    preferences = [[0], [1], [0,1], [0, 1], [1, 0], [1], [1, 0], [0, 1], [1]]
    licences = [1, 4, 0, 5, 8]
    print(allocate(preferences, licences))

