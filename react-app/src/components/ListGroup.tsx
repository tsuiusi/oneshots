// import { Fragment } from "react";
// Can be achieved by just using <> and not importing fragment


// a component cannot return more than 1 element (listgroup can only return listgroup and not h1)
// this can be solved by wrapping the whole thing in a div or fragments
function ListGroup() {
	const items = [
		'New York',
		'Hong Kong', 
		'London'
	];

	return (
		<>	
			<h1>List</h1>
			<ul className="list-group">
							{items.map((item) => (<li>{item}</li>))}
			</ul>
		</>
	);
}


export default ListGroup;
