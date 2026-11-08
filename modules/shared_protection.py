"""
Shared Path Protection Module for Elastic Optical Networks.

This module implements shared protection algorithms for elastic optical networks.
It supports shared spectrum allocation for protection paths, improving resource
utilization compared to 1+1 protection while maintaining resilience.

Features:
- Generation of shared protection requests from traffic matrix
- Shared spectrum allocation algorithm
- Simulated annealing optimization with shared protection
- Metrics for evaluating shared protection effectiveness
"""

import random
import time
import math
import numpy as np
from .modulation import choose_modulation, MODULATION_FORMATS
from .metrics import get_all_metrics
import networkx as nx
from copy import deepcopy
from collections import defaultdict
from modules.optimization_RMSA import (
    Link, Network, Request, get_edges_from_path,
    check_disjoint_paths, find_disjoint_path_pairs
)


def compute_path_length(G, path):
    """
    Compute the physical length of a path.

    Args:
        G: NetworkX graph representing the topology
        path: List of nodes representing the path

    Returns:
        float: The physical length of the path
    """
    return sum(G[u][v]['weight'] for u, v in zip(path, path[1:]))


def compute_modulation_format(path_length):
    """
    Determine the best modulation format for a given path length.

    Args:
        path_length: Physical length of the path

    Returns:
        str: The name of the selected modulation format
    """
    best_mod = None
    max_rate = 0

    for mod, props in MODULATION_FORMATS.items():
        if path_length <= props["max_length"] and props["rate"] > max_rate:
            max_rate = props["rate"]
            best_mod = mod

    return best_mod


def compute_required_slots(demand_gbps, modulation_format):
    """
    Compute the number of slots required for a demand with a specific modulation.

    Args:
        demand_gbps: Bandwidth demand in Gbps
        modulation_format: Selected modulation format

    Returns:
        int: Number of slots required
    """
    if modulation_format not in MODULATION_FORMATS:
        return None

    props = MODULATION_FORMATS[modulation_format]
    rate = props["rate"]
    slots_per_lightpath = props["slots"]

    # Calculate number of lightpaths needed (ceiling division)
    num_lightpaths = -(-demand_gbps // rate)
    total_slots = int(slots_per_lightpath * num_lightpaths)

    return total_slots


class SharedProtectedRequest:
    """
    Class representing a request with shared protection path.

    This extends the concept of protected requests by allowing sharing
    of spectrum resources on protection paths when the corresponding
    working paths are disjoint.
    """
    def __init__(self, id, source, destination, demand, working_path=None, protection_path=None,
                 modulation_format=None, start_slot=None, end_slot=None,
                 protection_start_slot=None, protection_end_slot=None, physical_distance=None):
        """
        Initialize a shared protected request.

        Args:
            id: Request identifier
            source: Source node
            destination: Destination node
            demand: Bandwidth demand
            working_path: Primary path
            protection_path: Backup path
            modulation_format: Selected modulation format
            start_slot: Starting slot for primary path
            end_slot: Ending slot for primary path
            protection_start_slot: Starting slot for backup path
            protection_end_slot: Ending slot for backup path
            physical_distance: Physical distance of the working path
        """
        self.id = id
        self.source = source
        self.destination = destination
        self.demand = demand
        self.working_path = working_path
        self.protection_path = protection_path
        self.modulation_format = modulation_format
        self.start_slot = start_slot
        self.end_slot = end_slot
        self.protection_start_slot = protection_start_slot
        self.protection_end_slot = protection_end_slot
        self.required_slots = None
        self.protection_required_slots = None
        self.transponder_cost = 0
        self.primary_path_group = None  # Add this for shared protection algorithm
        self.physical_distance = physical_distance  # 添加物理距离属性


def generate_shared_requests_from_matrix(G, traffic_matrix, k=3, order="ascending"):
    """
    Generate shared protection requests from traffic matrix.

    Args:
        G: NetworkX graph representing the topology
        traffic_matrix: Traffic demand matrix (list of tuples (source, destination, demand))
        k: Number of k-shortest paths to compute
        order: Order to process demands ("ascending" or "descending")

    Returns:
        list: List of SharedProtectedRequest objects
    """
    # Generate all pairs of shortest paths
    all_pairs_paths = {}
    nodes = list(G.nodes())

    for source in nodes:
        for destination in nodes:
            if source != destination:
                # Use k-shortest paths
                paths = list(nx.shortest_simple_paths(G, source, destination))
                all_pairs_paths[(source, destination)] = paths[:k]

    # Create requests from traffic matrix
    requests = []
    request_id = 0

    # Sort the traffic matrix based on specified order
    sorted_traffic = sorted(traffic_matrix, key=lambda x: x[2], reverse=(order == "descending"))

    # Convert demands to requests
    for source, destination, demand in sorted_traffic:
        # Get primary and backup paths
        if (source, destination) in all_pairs_paths and len(all_pairs_paths[(source, destination)]) >= 2:
            primary_path = all_pairs_paths[(source, destination)][0]
            backup_path = all_pairs_paths[(source, destination)][1]

            # Ensure backup path is link-disjoint from primary
            is_disjoint = True
            for i in range(len(primary_path) - 1):
                primary_link = (primary_path[i], primary_path[i+1])
                for j in range(len(backup_path) - 1):
                    backup_link = (backup_path[j], backup_path[j+1])
                    if primary_link == backup_link or (primary_link[1], primary_link[0]) == backup_link:
                        is_disjoint = False
                        break
                if not is_disjoint:
                    break

            if not is_disjoint and len(all_pairs_paths[(source, destination)]) > 2:
                # Try the third path if available
                backup_path = all_pairs_paths[(source, destination)][2]

                # Check disjointness again
                is_disjoint = True
                for i in range(len(primary_path) - 1):
                    primary_link = (primary_path[i], primary_path[i+1])
                    for j in range(len(backup_path) - 1):
                        backup_link = (backup_path[j], backup_path[j+1])
                        if primary_link == backup_link or (primary_link[1], primary_link[0]) == backup_link:
                            is_disjoint = False
                            break
                    if not is_disjoint:
                        break

            # Calculate modulation format and required slots
            primary_path_length = compute_path_length(G, primary_path)
            modulation_format = compute_modulation_format(primary_path_length)
            required_slots = compute_required_slots(demand, modulation_format)

            backup_path_length = compute_path_length(G, backup_path)
            backup_modulation_format = compute_modulation_format(backup_path_length)
            backup_required_slots = compute_required_slots(demand, backup_modulation_format)

            # Create the request
            request = SharedProtectedRequest(
                id=request_id,
                source=source,
                destination=destination,
                demand=demand,
                working_path=primary_path,
                protection_path=backup_path if is_disjoint else None,
                modulation_format=modulation_format,
                physical_distance=primary_path_length  # 设置物理距离
            )
            request.required_slots = required_slots
            request.protection_required_slots = backup_required_slots

            # Calculate transponder cost
            if modulation_format == "DP-16QAM":
                request.transponder_cost = 3.7
            elif modulation_format == "SC-DP-16QAM":
                request.transponder_cost = 2.0
            elif modulation_format == "SC-DP-QPSK":
                request.transponder_cost = 1.5

            requests.append(request)
            request_id += 1

    return requests


class SharedProtectionNetwork(Network):
    """
    Extended Network class that supports shared protection path resource allocation.
    """

    def __init__(self, graph):
        super().__init__(graph)
        # Track which primary paths use each link
        self.link_to_primary_paths = defaultdict(set)  # {link: {request_ids}}
        # Track backup resources that can be shared
        self.backup_resource_map = {}  # {link: {slot: {primary_path_groups}}}

    def primary_path_group(self, path):
        """Get a hashable representation of a path for grouping"""
        return frozenset(get_edges_from_path(path))

    def calculate_required_slots(self, demand_gbps, modulation_format):
        """Wrapper for the compute_required_slots function"""
        return compute_required_slots(demand_gbps, modulation_format)

    def get_link(self, u, v):
        """Get a link object for a specific edge, create if not exists"""
        if (u, v) not in self.links:
            self.links[(u, v)] = Link(u, v)
        return self.links[(u, v)]

    def are_sharable(self, path1, path2):
        """
        Check if two primary paths can share backup resources.

        Improved version with more flexible sharing criteria:
        1. Uses risk groups instead of direct path overlap
        2. Allows more sharing opportunities with higher threshold
        3. Considers the network topology when determining if paths can share

        Args:
            path1: First primary path
            path2: Second primary path

        Returns:
            bool: True if the paths can share backup resources
        """
        # Convert paths to sets of edges
        edges1 = set(get_edges_from_path(path1))
        edges2 = set(get_edges_from_path(path2))

        # Calculate overlap
        common_edges = edges1.intersection(edges2)

        # If no common edges, paths are completely disjoint - always allow sharing
        if not common_edges:
            return True

        # For paths with some overlap, use more flexible criteria
        total_unique_edges = len(edges1.union(edges2))

        # More permissive overlap threshold (increasing from 30% to 50%)
        # This allows more sharing opportunities
        MAX_OVERLAP_RATIO = 0.5
        overlap_ratio = len(common_edges) / total_unique_edges if total_unique_edges > 0 else 0

        # Return True if overlap is acceptable
        return overlap_ratio <= MAX_OVERLAP_RATIO

    def assign_shared_spectrum(self, request, modulation):
        """
        Try to assign spectrum for both primary and backup paths of a shared protected request.

        Args:
            request: The SharedProtectedRequest object
            modulation: Modulation format

        Returns:
            bool: True if spectrum assignment was successful, False otherwise
        """
        # Skip if no working or protection path
        if not request.working_path or not request.protection_path:
            return False

        # Calculate primary path group (set of edges in the primary path)
        primary_path = request.working_path
        # 使用frozenset而不是set，确保primary_path_group是可哈希的
        primary_path_group = frozenset(get_edges_from_path(primary_path))
        request.primary_path_group = primary_path_group

        # Get required slots based on modulation
        required_slots = request.required_slots

        # First try to allocate primary path
        primary_edges = get_edges_from_path(primary_path)

        # Check if all links have enough free slots for primary path
        primary_available_slots = []
        for edge in primary_edges:
            u, v = edge
            link = self.get_link(u, v)
            link_available = []

            # Find all possible starting slots on this link
            for i in range(320 - required_slots + 1):
                all_free = True
                for j in range(required_slots):
                    if i + j >= len(link.slots) or link.slots[i + j] == 1:
                        all_free = False
                        break
                if all_free:
                    link_available.append(i)

            primary_available_slots.append(link_available)

        if not all(primary_available_slots):
            return False  # No available slots on at least one link

        # Find start slot that's available on all links
        primary_valid_slots = set(primary_available_slots[0])
        for slots in primary_available_slots[1:]:
            primary_valid_slots.intersection_update(slots)

        if not primary_valid_slots:
            return False

        # Choose lowest index slot for primary path
        primary_start = min(primary_valid_slots)

        # Try to allocate backup path, potentially sharing resources
        backup_path = request.protection_path
        backup_edges = get_edges_from_path(backup_path)
        backup_required_slots = request.protection_required_slots

        # Check if we can share backup resources
        backup_start, sharing_percentage = self.find_sharable_backup_slots(backup_edges, backup_required_slots, primary_path_group)

        if backup_start is None:
            # If sharing not possible, try to allocate fresh resources
            backup_available_slots = []
            for edge in backup_edges:
                u, v = edge
                link = self.get_link(u, v)
                link_available = []

                # Find all possible starting slots on this link
                for i in range(320 - backup_required_slots + 1):
                    all_free = True
                    for j in range(backup_required_slots):
                        if i + j >= len(link.slots) or link.slots[i + j] == 1:
                            all_free = False
                            break
                    if all_free:
                        link_available.append(i)

                backup_available_slots.append(link_available)

            if not all(backup_available_slots):
                return False  # No available slots on at least one link

            # Find start slot that's available on all links
            backup_valid_slots = set(backup_available_slots[0])
            for slots in backup_available_slots[1:]:
                backup_valid_slots.intersection_update(slots)

            if not backup_valid_slots:
                return False

            # Choose lowest index slot for backup path
            backup_start = min(backup_valid_slots)
            sharing_percentage = 0

        # Success! Update request with allocation details
        request.start_slot = primary_start
        request.end_slot = primary_start + required_slots - 1
        request.protection_start_slot = backup_start
        request.protection_end_slot = backup_start + backup_required_slots - 1

        # Track sharing statistics for this request
        request.sharing_percentage = sharing_percentage

        # Update link to primary paths mapping
        for edge in primary_edges:
            self.link_to_primary_paths[edge].add(request.id)

        # Permanent assignment for primary path
        for edge in primary_edges:
            u, v = edge
            link = self.get_link(u, v)
            link.assign_slots(primary_start, required_slots)

        # Update backup resource map and assign slots
        self.assign_shared_backup_resources(
            backup_path, backup_start, backup_required_slots, primary_path_group)

        return True

    def assign_shared_spectrum_on_backup(self, path, required_slots, primary_path_group):
        """
        Try to assign spectrum on a backup path, potentially sharing resources.

        Improved version:
        1. Handles the new sharing mechanism with sharing percentage information
        2. Prioritizes sharing over new resource allocation
        3. Includes sharing statistics

        Args:
            path: Backup path
            required_slots: Number of slots needed
            primary_path_group: Set of links in the primary path

        Returns:
            tuple: (start_slot, sharing_percentage) or (None, 0) if assignment fails
        """
        # Get all edges in the path
        edges = get_edges_from_path(path)

        # Check if we can share existing backup resources
        best_shared_start, sharing_percentage = self.find_sharable_backup_slots(edges, required_slots, primary_path_group)

        if best_shared_start is not None:
            return best_shared_start, sharing_percentage

        # Try to allocate new resources if sharing is not possible
        # Check if all links have enough free slots
        available_slots = []
        for edge in edges:
            u, v = edge
            link = self.get_link(u, v)
            link_available = []

            # Find all possible starting slots on this link
            for i in range(320 - required_slots + 1):
                all_free = True
                for j in range(required_slots):
                    if i + j >= len(link.slots) or link.slots[i + j] == 1:
                        all_free = False
                        break
                if all_free:
                    link_available.append(i)

            available_slots.append(link_available)

        if not all(available_slots):
            return None, 0  # No available slots on at least one link

        # Find start slot that's available on all links (least used strategy)
        slot_usage = defaultdict(int)
        for slots in available_slots:
            for s in slots:
                slot_usage[s] += 1

        # Find slots that are available on all links
        valid_slots = [s for s in slot_usage if slot_usage[s] == len(edges)]
        if not valid_slots:
            return None, 0

        # Choose lowest index slot (for better spectrum utilization)
        start_slot = min(valid_slots)
        return start_slot, 0  # 0 sharing percentage for new allocation

    def find_sharable_backup_slots(self, edges, required_slots, primary_path_group):
        """
        Find slots that can be shared with existing backup resources.

        Enhanced version to maximize sharing opportunities:
        1. Prioritizes slots with maximum sharing potential
        2. Uses weighted scoring to favor slots with higher sharing
        3. Implements a more aggressive sharing strategy

        Args:
            edges: List of edges in the backup path
            required_slots: Number of slots needed
            primary_path_group: Set of links in the primary path (as frozenset)

        Returns:
            tuple: (start_slot, sharing_percentage) - Starting slot and percentage of shared resources
        """
        # Maximum allowable slot index
        max_slot = 320 - required_slots

        # Track sharing opportunities for each potential starting slot
        sharing_opportunities = {}

        # Keep track of the best sharing score for each potential starting slot
        sharing_scores = {}

        # For each potential starting slot
        for start_slot in range(max_slot + 1):
            # Track sharable edges and total sharing score for this start_slot
            sharable_edges = 0
            total_edges = len(edges)
            sharing_score = 0.0
            all_slots_available = True

            # Check each edge for sharing possibility
            for edge in edges:
                u, v = edge
                link = self.get_link(u, v)

                # Track if we can share/use all slots for this edge
                edge_slots_available = True
                edge_sharable = False
                edge_sharing_score = 0.0

                # Check all required slots for this edge
                for offset in range(required_slots):
                    slot_idx = start_slot + offset

                    # If slot is free, we can use it but not share (score it lower)
                    if link.slots[slot_idx] == 0:
                        continue

                    # If slot is used, check if it's a backup resource we can share
                    edge_key = (u, v)
                    if (edge_key in self.backup_resource_map and
                        slot_idx in self.backup_resource_map[edge_key]):

                        # Check if any existing primary path group conflicts with our primary path
                        existing_groups = self.backup_resource_map[edge_key][slot_idx]

                        # Convert primary_path_group to list for are_sharable function if needed
                        # Note: primary_path_group is already a frozenset from the caller
                        primary_path_list = list(primary_path_group)

                        # Check if we can share with all existing groups
                        can_share = True
                        # Count how many existing groups we can share with (more is better)
                        sharable_groups = 0

                        for group in existing_groups:
                            # Convert existing group to list too if it's a frozenset
                            existing_group_list = list(group) if isinstance(group, frozenset) else list(group)

                            if self.are_sharable(primary_path_list, existing_group_list):
                                sharable_groups += 1
                            else:
                                can_share = False
                                break

                        if can_share:
                            edge_sharable = True
                            # Score higher if we can share with more groups
                            edge_sharing_score += 0.2 * sharable_groups
                            continue

                    # If we reach here, the slot is not available for this edge
                    edge_slots_available = False
                    break

                # If all slots are available for this edge (either free or sharable)
                if edge_slots_available:
                    # Score higher for shared edges versus just free edges
                    if edge_sharable:
                        sharable_edges += 1
                        sharing_score += 1.0 + edge_sharing_score  # Bonus for sharing
                    else:
                        sharing_score += 0.1  # Small score for just being available
                else:
                    # If any edge doesn't have all slots available, this start slot won't work
                    all_slots_available = False
                    break

            # If all edges have their slots available
            if all_slots_available:
                # Calculate sharing percentage - how many edges can share existing backup resources
                sharing_percentage = sharable_edges / total_edges if total_edges > 0 else 0
                sharing_opportunities[start_slot] = sharing_percentage
                sharing_scores[start_slot] = sharing_score

        if sharing_opportunities:
            # First prioritize slots with any sharing at all
            sharing_slots = [s for s, p in sharing_opportunities.items() if p > 0]

            if sharing_slots:
                # Choose the slot with maximum sharing score (prioritize sharing over just being available)
                best_start_slot = max(sharing_slots, key=lambda x: sharing_scores[x])
            else:
                # If no sharing is possible, choose any available slot
                best_start_slot = min(sharing_opportunities.keys())

            return best_start_slot, sharing_opportunities[best_start_slot]

        return None, 0

    def assign_shared_backup_resources(self, path, start_slot, num_slots, primary_path_group):
        """
        Assign or update backup resources, tracking sharing information.

        Args:
            path: Backup path
            start_slot: Starting slot
            num_slots: Number of slots to assign
            primary_path_group: Set of links in the primary path
        """
        for edge in get_edges_from_path(path):
            u, v = edge
            link = self.get_link(u, v)

            # Check if slots are already assigned (shared)
            new_assignment = []
            for offset in range(num_slots):
                slot_idx = start_slot + offset

                # Set slot as used in the link's slot array
                if link.slots[slot_idx] == 0:
                    link.slots[slot_idx] = 1
                    new_assignment.append(slot_idx)

                # Update backup resource map
                edge_key = (u, v)
                if edge_key not in self.backup_resource_map:
                    self.backup_resource_map[edge_key] = {}

                if slot_idx not in self.backup_resource_map[edge_key]:
                    self.backup_resource_map[edge_key][slot_idx] = set()

                # Add our primary path group to the set
                self.backup_resource_map[edge_key][slot_idx].add(primary_path_group)

    def remove_shared_assignment(self, path, start_slot, num_slots, primary_path_group):
        """
        Remove a shared backup path assignment.

        Args:
            path: Backup path
            start_slot: Starting slot
            num_slots: Number of slots
            primary_path_group: Set of links in the primary path (as frozenset)
        """
        if start_slot is None:
            return

        # 确保primary_path_group是frozenset类型，以便可以在set操作中使用
        if not isinstance(primary_path_group, frozenset):
            primary_path_group = frozenset(primary_path_group)

        for edge in get_edges_from_path(path):
            u, v = edge
            link = self.get_link(u, v)
            edge_key = (u, v)

            for offset in range(num_slots):
                slot_idx = start_slot + offset

                # Remove primary path group from backup resource map
                if (edge_key in self.backup_resource_map and
                    slot_idx in self.backup_resource_map[edge_key]):

                    self.backup_resource_map[edge_key][slot_idx].discard(primary_path_group)

                    # If no more primary path groups use this backup resource, free the slot
                    if not self.backup_resource_map[edge_key][slot_idx]:
                        del self.backup_resource_map[edge_key][slot_idx]
                        link.slots[slot_idx] = 0

                    # Clean up empty dictionaries
                    if not self.backup_resource_map[edge_key]:
                        del self.backup_resource_map[edge_key]

    def swap_assignments(self, request1, request2):
        """
        Swap the spectrum assignments of two requests.
        Used during simulated annealing optimization.

        Args:
            request1, request2: SharedProtectedRequest objects

        Returns:
            Boolean indicating if swap was successful
        """
        if (request1.start_slot is None or request2.start_slot is None or
            request1.end_slot != request2.end_slot):
            return False

        # Store original assignments
        original_request1 = deepcopy(request1)
        original_request2 = deepcopy(request2)

        # Remove both assignments
        self.remove_assignment(request1.working_path, request1.start_slot, request1.end_slot - request1.start_slot + 1)
        self.remove_shared_assignment(
            request1.protection_path, request1.start_slot, request1.end_slot - request1.start_slot + 1, request1.primary_path_group)

        self.remove_assignment(request2.working_path, request2.start_slot, request2.end_slot - request2.start_slot + 1)
        self.remove_shared_assignment(
            request2.protection_path, request2.start_slot, request2.end_slot - request2.start_slot + 1, request2.primary_path_group)

        # Update link to primary paths mapping
        for edge in get_edges_from_path(request1.working_path):
            self.link_to_primary_paths[edge].discard(request1.id)

        for edge in get_edges_from_path(request2.working_path):
            self.link_to_primary_paths[edge].discard(request2.id)

        # Try to reassign with swapped start slots
        # First, swap primary paths
        swap_primary_path = request1.working_path
        swap_primary_start = request1.start_slot
        request1.working_path = request2.working_path
        request1.start_slot = request2.start_slot
        request2.working_path = swap_primary_path
        request2.start_slot = swap_primary_start

        # Then, swap backup paths
        swap_backup_path = request1.protection_path
        swap_backup_start = request1.start_slot
        request1.protection_path = request2.protection_path
        request1.start_slot = request2.start_slot
        request2.protection_path = swap_backup_path
        request2.start_slot = swap_backup_start

        # Update primary path groups
        request1.primary_path_group = self.primary_path_group(request1.working_path)
        request2.primary_path_group = self.primary_path_group(request2.working_path)

        # Try to reassign request1
        success1 = True
        for edge in get_edges_from_path(request1.working_path):
            u, v = edge
            link = self.get_link(u, v)
            if not all(link.slots[request1.start_slot + j] == 0 for j in range(request1.end_slot - request1.start_slot + 1)):
                success1 = False
                break

        # If primary path can be assigned, assign it and update mappings
        if success1:
            for edge in get_edges_from_path(request1.working_path):
                u, v = edge
                link = self.get_link(u, v)
                link.assign_slots(request1.start_slot, request1.end_slot - request1.start_slot + 1)
                self.link_to_primary_paths[edge].add(request1.id)

            # Try to assign or share backup path for request1
            backup_success = True
            self.assign_shared_backup_resources(
                request1.protection_path, request1.start_slot,
                request1.end_slot - request1.start_slot + 1, request1.primary_path_group)
        else:
            backup_success = False

        # Try to reassign request2
        success2 = True
        if success1 and backup_success:
            for edge in get_edges_from_path(request2.working_path):
                u, v = edge
                link = self.get_link(u, v)
                if not all(link.slots[request2.start_slot + j] == 0 for j in range(request2.end_slot - request2.start_slot + 1)):
                    success2 = False
                    break

            # If primary path can be assigned, assign it and update mappings
            if success2:
                for edge in get_edges_from_path(request2.working_path):
                    u, v = edge
                    link = self.get_link(u, v)
                    link.assign_slots(request2.start_slot, request2.end_slot - request2.start_slot + 1)
                    self.link_to_primary_paths[edge].add(request2.id)

                # Try to assign or share backup path for request2
                backup_success2 = True
                self.assign_shared_backup_resources(
                    request2.protection_path, request2.start_slot,
                    request2.end_slot - request2.start_slot + 1, request2.primary_path_group)
            else:
                backup_success2 = False
        else:
            success2 = False
            backup_success2 = False

        # If any reassignment failed, revert to original state
        if not (success1 and backup_success and success2 and backup_success2):
            # Revert request1
            request1.working_path = original_request1.working_path
            request1.start_slot = original_request1.start_slot
            request1.end_slot = original_request1.end_slot
            request1.protection_path = original_request1.protection_path
            request1.primary_path_group = original_request1.primary_path_group

            # Revert request2
            request2.working_path = original_request2.working_path
            request2.start_slot = original_request2.start_slot
            request2.end_slot = original_request2.end_slot
            request2.protection_path = original_request2.protection_path
            request2.primary_path_group = original_request2.primary_path_group

            # Clean up any partial assignments
            if success1:
                self.remove_assignment(request1.working_path, request1.start_slot, request1.end_slot - request1.start_slot + 1)
                if backup_success:
                    self.remove_shared_assignment(
                        request1.protection_path, request1.start_slot,
                        request1.end_slot - request1.start_slot + 1, request1.primary_path_group)

            if success2:
                self.remove_assignment(request2.working_path, request2.start_slot, request2.end_slot - request2.start_slot + 1)
                if backup_success2:
                    self.remove_shared_assignment(
                        request2.protection_path, request2.start_slot,
                        request2.end_slot - request2.start_slot + 1, request2.primary_path_group)

            # Restore original assignments
            for edge in get_edges_from_path(original_request1.working_path):
                u, v = edge
                link = self.get_link(u, v)
                link.assign_slots(original_request1.start_slot, original_request1.end_slot - original_request1.start_slot + 1)
                self.link_to_primary_paths[edge].add(original_request1.id)

            self.assign_shared_backup_resources(
                original_request1.protection_path, original_request1.start_slot,
                original_request1.end_slot - original_request1.start_slot + 1, original_request1.primary_path_group)

            for edge in get_edges_from_path(original_request2.working_path):
                u, v = edge
                link = self.get_link(u, v)
                link.assign_slots(original_request2.start_slot, original_request2.end_slot - original_request2.start_slot + 1)
                self.link_to_primary_paths[edge].add(original_request2.id)

            self.assign_shared_backup_resources(
                original_request2.protection_path, original_request2.start_slot,
                original_request2.end_slot - original_request2.start_slot + 1, original_request2.primary_path_group)

            return False

        return True


def simulated_annealing_shared_protection(shared_requests, network, max_iter=10000, initial_T=1.0, cooling_rate=0.97):
    """
    Simulated annealing optimization for network with shared protection.

    Enhanced version with improved sharing efficiency:
    1. Prioritizes moves that increase sharing ratio
    2. Uses more sophisticated acceptance criteria for sharing
    3. Return complete solution including best network state

    Args:
        shared_requests: List of SharedProtectedRequest objects
        network: SharedProtectionNetwork object
        max_iter: Maximum number of iterations
        initial_T: Initial temperature
        cooling_rate: Cooling rate

    Returns:
        tuple: (best_solution, best_requests, best_network, best_frag) - Best solution found
    """
    # Make a deep copy of the current network and requests
    current_network = deepcopy(network)
    current_requests = deepcopy(shared_requests)

    # Create an initial solution - dictionary mapping request ID to spectrum allocation
    solution = {}
    for req in current_requests:
        solution[req.id] = (req.start_slot, req.protection_start_slot)

    # Record current fragmentation and network state
    current_frag = current_network.compute_network_fragmentation()

    # Track best solution
    best_solution = deepcopy(solution)
    best_frag = current_frag
    best_network = deepcopy(current_network)
    best_requests = deepcopy(current_requests)

    # Calculate initial sharing ratio
    total_actual_backup_slots = sum(
        req.protection_end_slot - req.protection_start_slot + 1
        for req in current_requests
        if req.protection_start_slot is not None
    )
    total_required_backup_slots = sum(
        req.protection_required_slots
        for req in current_requests
        if req.protection_required_slots is not None
    )
    current_sharing_ratio = total_required_backup_slots / total_actual_backup_slots if total_actual_backup_slots > 0 else 1.0
    best_sharing_ratio = current_sharing_ratio

    # Initialize
    T = initial_T
    improved_count = 0

    print(f"初始解生成完成，开始模拟退火优化(带共享保护)，最大迭代次数: {max_iter}...")
    progress_step = max(1, max_iter // 10)

    # Main simulated annealing loop
    for it in range(max_iter):
        # Print progress
        if (it + 1) % progress_step == 0 or it == 0:
            print(f"模拟退火进度: {(it + 1) / max_iter * 100:.1f}%, 当前温度: {T:.6f}, 当前最佳碎片化: {best_frag:.6f}")

        # Stop if temperature becomes too small
        if T < 1e-8:
            print(f"温度已接近0，提前结束模拟退火，完成迭代次数: {it+1}/{max_iter}")
            break

        # Select two random requests for swapping
        if len(current_requests) < 2:
            break

        req_idx1, req_idx2 = random.sample(range(len(current_requests)), 2)
        req1, req2 = current_requests[req_idx1], current_requests[req_idx2]

        # Create a temporary network copy for trying the swap
        temp_network = deepcopy(current_network)

        # Try to swap the requests
        success = temp_network.swap_assignments(req1, req2)

        if success:
            # Calculate new fragmentation and sharing ratio
            new_frag = temp_network.compute_network_fragmentation()

            # Calculate new sharing ratio
            new_total_actual_backup_slots = sum(
                req.protection_end_slot - req.protection_start_slot + 1
                for req in [req1, req2]
                if req.protection_start_slot is not None
            )
            new_total_required_backup_slots = sum(
                req.protection_required_slots
                for req in [req1, req2]
                if req.protection_required_slots is not None
            )

            # Get previous slots for these requests
            prev_actual_backup_slots = sum(
                current_requests[i].protection_end_slot - current_requests[i].protection_start_slot + 1
                for i in [req_idx1, req_idx2]
                if current_requests[i].protection_start_slot is not None
            )

            # Update the total for sharing ratio calculation
            new_sharing_ratio = current_sharing_ratio  # Default to current

            # Only if the backup slots have changed, recalculate the ratio
            if new_total_actual_backup_slots != prev_actual_backup_slots:
                adjusted_total_actual = total_actual_backup_slots - prev_actual_backup_slots + new_total_actual_backup_slots
                if adjusted_total_actual > 0:
                    new_sharing_ratio = total_required_backup_slots / adjusted_total_actual

            # Calculate energy difference - consider both fragmentation and sharing ratio
            # We want to minimize fragmentation and maximize sharing ratio
            frag_diff = new_frag - current_frag
            sharing_diff = current_sharing_ratio - new_sharing_ratio  # Negative if sharing improves

            # Combined energy difference (weighted sum - prioritize sharing)
            energy_diff = frag_diff - 2.0 * sharing_diff  # We prioritize sharing with higher weight

            # Decide whether to accept the new solution
            accept = False
            if energy_diff <= 0:  # Better solution
                accept = True
            else:
                # Accept with probability based on temperature
                p = math.exp(-energy_diff / T)
                accept = random.random() < p

            if accept:
                # Update current solution
                current_network = temp_network
                current_frag = new_frag
                current_sharing_ratio = new_sharing_ratio
                solution[req1.id] = (req1.start_slot, req1.protection_start_slot)
                solution[req2.id] = (req2.start_slot, req2.protection_start_slot)

                # Update requests with new slot assignments
                current_requests[req_idx1] = deepcopy(req1)
                current_requests[req_idx2] = deepcopy(req2)

                # Check if this is the best solution so far (prioritize sharing if close in fragmentation)
                if new_frag < best_frag or (abs(new_frag - best_frag) < 0.01 and new_sharing_ratio > best_sharing_ratio):
                    best_solution = deepcopy(solution)
                    best_frag = new_frag
                    best_network = deepcopy(current_network)
                    best_requests = deepcopy(current_requests)
                    best_sharing_ratio = new_sharing_ratio
                    improved_count += 1

        # Decrease temperature
        T *= cooling_rate

    print(f"✅ 模拟退火完成(带共享保护)：最佳碎片化度量 = {best_frag}，总改进次数 = {improved_count}")

    return best_solution, best_requests, best_network, best_frag


def run_sa_with_shared_protection(G, traffic_matrix, benchmark_metrics=None, order="ascending",
                                 max_iterations=10000, initial_temp=1.0, cooling_rate=0.97):
    """
    Run simulated annealing with shared protection.

    Enhanced version with improved sharing strategy:
    1. Calculates more accurate sharing metrics
    2. Uses improved path selection strategy
    3. Prioritizes sharing opportunities in resource allocation

    Args:
        G: NetworkX graph representing the topology
        traffic_matrix: Traffic demand matrix
        benchmark_metrics: Benchmark algorithm metrics for comparison
        order: Order to process demands ("ascending" or "descending")
        max_iterations: Maximum number of SA iterations
        initial_temp: Initial temperature for SA
        cooling_rate: Cooling rate for SA

    Returns:
        tuple: (metrics, slot_grid) - Performance metrics and spectrum allocation
    """
    print(f"\n运行共享保护算法 ({order})...")

    # Generate requests from traffic matrix
    shared_requests = generate_shared_requests_from_matrix(G, traffic_matrix, k=3, order=order)
    print(f"生成了 {len(shared_requests)} 个请求，包括并行请求")

    # Initialize network
    network = SharedProtectionNetwork(G)

    # Count total backup slots that would be needed without sharing
    theoretical_total_backup_slots = 0

    # Track successful requests
    successful_requests = []
    failed_requests = []

    # Initialize solution - initial assignment of slots
    print("正在生成初始解(带共享保护)...")
    for req in shared_requests:
        if not req.working_path or not req.protection_path:
            continue

        # Try to assign spectrum using the shared protection mechanism
        success = network.assign_shared_spectrum(req, req.modulation_format)

        if success:
            # Calculate theoretical backup slots (for sharing ratio)
            backup_slots_needed = req.protection_end_slot - req.protection_start_slot + 1
            theoretical_total_backup_slots += backup_slots_needed

            successful_requests.append(req)
        else:
            print(f"请求 {req.id} 在主路径 {req.working_path} 上分配频谱失败")
            failed_requests.append(req)

    if not successful_requests:
        print("⚠️ 所有请求均分配失败，无法进行模拟退火优化。")
        return {}, None

    # Run simulated annealing
    best_solution, best_requests, best_network, best_frag = simulated_annealing_shared_protection(
        successful_requests, network, max_iter=max_iterations,
        initial_T=initial_temp, cooling_rate=cooling_rate
    )

    # Print metrics
    print("\n共享保护算法性能指标:")
    print(f"成功分配请求: {len(successful_requests)} 个, 总共 {len(shared_requests)} 个请求")

    # Get the slot grid for visualization
    slot_grid = {}
    for edge, link in best_network.links.items():
        slot_grid[edge] = link.slots

    # Calculate metrics
    shared_metrics = get_all_metrics(slot_grid, successful_requests)

    # Calculate total slots used for backup paths (actual)
    total_actual_backup_slots = 0
    total_theoretical_backup_slots = 0
    total_sharing_percentage = 0
    shared_requests_count = 0

    for req in successful_requests:
        if hasattr(req, 'protection_start_slot') and req.protection_start_slot is not None:
            # Calculate actual backup slots used
            backup_slots_used = req.protection_end_slot - req.protection_start_slot + 1
            total_actual_backup_slots += backup_slots_used

            # Calculate theoretical backup slots that would be needed without sharing
            # This is the same as protection_required_slots
            theoretical_backup_slots = req.protection_required_slots
            total_theoretical_backup_slots += theoretical_backup_slots

            # Calculate sharing percentage (if available)
            if hasattr(req, 'sharing_percentage') and req.sharing_percentage > 0:
                total_sharing_percentage += req.sharing_percentage
                shared_requests_count += 1

    # Calculate sharing ratio - how many theoretical backup slots are effectively getting from actual backup slots
    # This should be > 1.0 for effective sharing (higher is better)
    sharing_ratio = total_theoretical_backup_slots / max(1, total_actual_backup_slots)

    # Calculate average sharing percentage
    avg_sharing_percentage = total_sharing_percentage / max(1, len(successful_requests))

    # Calculate shared requests percentage
    shared_requests_percentage = (shared_requests_count / max(1, len(successful_requests)) * 100)

    # Calculate resource savings compared to 1+1 protection
    resource_savings = 0
    if benchmark_metrics and 'total_used_fsus' in benchmark_metrics:
        # Calculate what 1+1 protection would use for these same successful requests
        # Primary path resources + backup path resources without sharing
        theoretical_1plus1_usage = sum(req.end_slot - req.start_slot + 1 for req in successful_requests) + \
                                  total_theoretical_backup_slots

        # Calculate actual resource usage with sharing
        actual_usage = shared_metrics['total_used_fsus']

        # Calculate saving percentage - positive means we saved resources
        if theoretical_1plus1_usage > 0:
            resource_savings = ((theoretical_1plus1_usage - actual_usage) / theoretical_1plus1_usage) * 100

    # Add sharing metrics to the metrics dictionary
    shared_metrics['sharing_ratio'] = sharing_ratio
    shared_metrics['avg_sharing_percentage'] = avg_sharing_percentage
    shared_metrics['shared_requests_count'] = shared_requests_count
    shared_metrics['shared_requests_percentage'] = shared_requests_percentage
    shared_metrics['resource_savings'] = resource_savings

    # Print additional metrics specific to shared protection
    print(f"最高使用的FSU: {shared_metrics['highest_used_fsu']}")
    print(f"总FSU使用量: {shared_metrics['total_used_fsus']}")
    print(f"平均链路利用率: {shared_metrics['avg_link_usage']:.4f}")
    print(f"频谱利用率熵: {shared_metrics['utilization_entropy']:.4f}")
    print(f"Shannon熵: {shared_metrics['shannon_entropy']:.4f}")
    print(f"共享比率: {sharing_ratio:.4f}")
    print(f"平均共享百分比: {avg_sharing_percentage:.4f}")
    print(f"共享资源的请求数量: {shared_requests_count} ({shared_requests_percentage:.2f}%)")
    print(f"资源节约率: {resource_savings:.2f}%")

    # Print comparison with benchmark algorithm (if provided)
    if benchmark_metrics:
        print("\n与基准算法对比:")
        success_rate = len(successful_requests) / len(shared_requests)
        print(f"共享保护算法成功率: {success_rate:.4f}")

        if 'max_fragmentation' in benchmark_metrics:
            print(f"基准算法频谱碎片: {benchmark_metrics['max_fragmentation']:.4f}")
            print(f"共享保护算法频谱碎片: {best_frag:.4f}")

        if 'total_used_fsus' in benchmark_metrics:
            print(f"基准算法FSU使用量: {benchmark_metrics['total_used_fsus']}")
            print(f"共享保护算法FSU使用量: {shared_metrics['total_used_fsus']}")

        if 'avg_link_usage' in benchmark_metrics:
            print(f"基准算法平均链路利用率: {benchmark_metrics['avg_link_usage']:.4f}")
            print(f"共享保护算法平均链路利用率: {shared_metrics['avg_link_usage']:.4f}")

    return shared_metrics, slot_grid